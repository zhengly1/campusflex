"""Experiment-layer guards: freeze gate, LLM preflight / fallback handling, Eq. 4-ordered
comparisons, E3 repeat pairing, evidence-ledger checks and the regenerated tables."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from dataclasses import replace

import pytest

from campusflex import __main__ as cli
from campusflex.config import HELDOUT_SEEDS, PlannerConfig
from campusflex.experiments import common, e1, e2, e3, e4, e6, evidence_ledger, freeze, tables
from campusflex.llm.client import ScriptedClient


def dead_client() -> ScriptedClient:
    """An LLM that never answers (as after a rate limit or network loss)."""
    return ScriptedClient(lambda role, system, user: RuntimeError("connection refused"))


@pytest.fixture
def no_frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "FROZEN", tmp_path / "missing_frozen_config.json")
    return tmp_path


# ------------------------------------------------------------------ freeze gate
def test_frozen_config_missing_present_and_tampered(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "FROZEN", tmp_path / "frozen_config.json")
    cfg, h, frozen = common.frozen_config()
    assert frozen is False and h == common.config_hash(PlannerConfig())
    rec = freeze.freeze(tmp_path / "frozen_config.json")
    assert rec["config_hash"] == h
    cfg2, h2, frozen2 = common.frozen_config()
    assert frozen2 is True and h2 == h and cfg2 == cfg
    with pytest.raises(FileExistsError):
        freeze.freeze(tmp_path / "frozen_config.json")
    freeze.freeze(tmp_path / "frozen_config.json", force=True)
    d = json.loads((tmp_path / "frozen_config.json").read_text())
    d["config"]["lambda_Omega"] = 8.0                       # edited after the gate
    (tmp_path / "frozen_config.json").write_text(json.dumps(d))
    with pytest.raises(ValueError):
        common.frozen_config()


def test_freeze_command_registered(tmp_path):
    assert cli.COMMANDS["freeze"] == "campusflex.experiments.freeze"
    freeze.main(["--out", str(tmp_path / "f.json")])
    assert json.loads((tmp_path / "f.json").read_text())["config_hash"] == common.config_hash(PlannerConfig())
    with pytest.raises(SystemExit):
        freeze.main(["--out", str(tmp_path / "f.json")])


def test_freeze_locks_prompts_model_and_notices(tmp_path, monkeypatch):
    """Figure 4: the gate locks prompts and held-out notices, not only the weights."""
    from campusflex.llm import prompts
    from campusflex.roles import l2_proposer

    f = tmp_path / "frozen_config.json"
    monkeypatch.setattr(common, "FROZEN", f)
    m = common.llm_freeze_meta(ScriptedClient({}, model="deepseek-chat"))
    assert m["prompts_frozen"] is False and "freeze" in m["prompts_note"]
    rec = freeze.freeze(f)
    assert set(rec["llm"]["prompt_hashes"]) == {"L1", "L2", "L3", "L4", "L4_explain", "L5"}
    assert rec["notices_hash"] == prompts.notices_hash()
    client = ScriptedClient({}, model=rec["llm"]["model"])
    assert common.llm_freeze_meta(client)["prompts_frozen"] is True
    assert common.notices_freeze_meta()["notices_frozen"] is True
    m = common.llm_freeze_meta(ScriptedClient({}, model="other-model"))       # model switched after the gate
    assert m["prompts_frozen"] is False and "other-model" in m["prompts_note"]
    monkeypatch.setattr(l2_proposer, "L2_SYSTEM", l2_proposer.L2_SYSTEM + " Be brief.")   # prompt edited
    m = common.llm_freeze_meta(client)
    assert m["prompts_frozen"] is False and "L2" in m["prompts_note"] and "L1" not in m["prompts_note"]
    corpus = json.loads(prompts.NOTICE_CORPUS.read_text())
    corpus["heldout"] = {**corpus["heldout"], "_rewritten": True}             # notices rewritten after it
    alt = tmp_path / "corpus.json"
    alt.write_text(json.dumps(corpus))
    monkeypatch.setattr(prompts, "NOTICE_CORPUS", alt)
    assert common.notices_freeze_meta()["notices_frozen"] is False


def test_ledger_flags_prompts_changed_after_freeze(tmp_path):
    (tmp_path / "results").mkdir()
    meta = {"experiment": "trace T1", "frozen": True, "prompts_frozen": True, "notices_frozen": True, "seed": 10}
    t1 = tmp_path / "results" / "trace_T1.json"
    t1.write_text(json.dumps({"meta": meta, "complete": True}))
    assert evidence_ledger.ledger(tmp_path)[6]["status"] == "Executed"
    t1.write_text(json.dumps({"meta": {**meta, "prompts_frozen": False,
                                       "prompts_note": "prompts changed after the freeze gate: L2"},
                              "complete": True}))
    r = evidence_ledger.ledger(tmp_path)[6]
    assert r["status"] == "Partial" and "L2" in r["evidence"]


# ------------------------------------------------------------------ shared helpers
def test_llm_arm_stats_threshold():
    assert common.llm_arm_stats(10, 5)["status"] == "run"                 # exactly at the limit
    s = common.llm_arm_stats(10, 1, n_fallbacks=6, n_proposals=9, n_valid=3)
    assert s["status"] == "invalid_mostly_fallback" and s["fallback_rate"] == 0.6
    assert s["valid_proposal_rate"] == pytest.approx(1 / 3)
    assert common.llm_arm_stats(0, 0)["status"] == "invalid_no_llm_calls"


def test_llm_client_uses_preflight(monkeypatch):
    seen = {}

    def fake(cache=True, kind=None, **kw):
        seen.update(cache=cache, kind=kind, **kw)
        return None, "preflight failed: timeout"

    monkeypatch.setattr("campusflex.llm.client.live_client_or_reason", fake)
    client, reason = e2.llm_client_or_reason(argparse.Namespace(llm=True, provider="deepseek", model="m",
                                                                no_cache=False), cache=False)
    assert client is None and reason == "preflight failed: timeout"
    assert seen == {"cache": False, "kind": "deepseek", "model": "m"}
    assert e2.llm_client_or_reason(argparse.Namespace(llm=False))[0] is None


def test_lexicographic_compare_puts_unserved_first():
    tie_u = [0.0] * 10
    c = common.lexicographic_compare(tie_u, tie_u, [1.0] * 10, [2.0 + 0.01 * i for i in range(10)])
    assert c["decided_by"] == "score" and c["a_better"]
    # fewer unserved person-minutes wins even with a worse score
    c = common.lexicographic_compare([0.0] * 10, [30.0 + i for i in range(10)], [9.0] * 10, [1.0] * 10)
    assert c["decided_by"] == "unserved" and c["a_better"] and not c["a_worse"]
    assert c["excludes_zero"] and c["mean_diff"] == 8.0                # score bootstrap still reported


# ------------------------------------------------------------------ E1
def _e1_row(arm, scen, seed, score, h, calls=10, fallbacks=0, unserved=0):
    return {"arm": arm, "policy": arm[-1], "scenario": scen, "seed": seed, "mismatch": False, "repeat": 0,
            "score": score, "unserved_person_min": unserved, "config_hash": h, "llm_l2_calls": calls,
            "llm_l2_fallbacks": fallbacks, "llm_n_failed_calls": fallbacks, "llm_l2_plans": 5, "llm_l2_valid": 4}


def test_e1_expected_episodes_match_protocol():
    assert e1.expected_episodes("AB", False) == 160 and e1.expected_episodes("CD", False) == 480
    assert e1.expected_episodes("AB", True) + e1.expected_episodes("CD", True) == 80


def test_e1_analyse_refuses_config_mismatch_and_reports_meta():
    ab = {"meta": {"config_hash": "aaa", "reduced": False, "frozen": True},
          "rows": [_e1_row(a, "fault", s, 10.0 + s, "aaa") for a in ("arm_A", "arm_B") for s in range(10, 20)]}
    cd = {"meta": {"config_hash": "bbb", "reduced": True, "frozen": True},
          "rows": [_e1_row(a, "fault", s, 5.0 + s, "bbb") for a in ("arm_C", "arm_D") for s in range(10, 20)]}
    an = e1.analyse(ab, cd)
    assert an["comparisons"]["arm_B-arm_A"]["status"] == "computed"
    for k in ("arm_C-arm_B", "arm_D-arm_B", "arm_C-arm_A"):
        assert an["comparisons"][k]["status"] == "refused_config_mismatch"
    assert an["success_criterion_met"] is None
    assert an["meta"]["reduced"] is True and an["meta"]["CD"]["config_hash"] == "bbb"
    # same configuration -> compared; C clearly better than B
    for r in cd["rows"]:
        r["config_hash"] = "aaa"
    cd["meta"]["config_hash"] = "aaa"
    an = e1.analyse(ab, cd)
    assert an["comparisons"]["arm_C-arm_B"]["status"] == "computed"
    assert an["success_criterion_met"] is True and "arm_C-arm_B" in an["winning_arms"]
    per = {r["arm"]: r for r in an["per_arm"]}
    assert per["arm_C"]["llm"]["fallback_rate"] == 0.0 and per["arm_C"]["llm"]["valid_proposal_rate"] == 0.8


def test_e1_cd_dead_llm_is_invalid_and_traced(no_frozen, tmp_path):
    out = tmp_path / "e1_CD.json"
    cd = e1.run_CD(dead_client, reduced=True, seeds=(10,), threads=2, out=str(out))
    meta = cd["meta"]
    assert meta["frozen"] is False and meta["expected_episodes"] == 4 and len(cd["rows"]) == 4
    assert {s["status"] for s in meta["llm_arm_stats"].values()} == {"invalid_mostly_fallback"}
    assert all(r["llm_fallback_rate"] == 1.0 and r["llm_n_failed_calls"] == r["llm_l2_calls"] for r in cd["rows"])
    with gzip.open(meta["traces"], "rt") as f:
        traces = [json.loads(line) for line in f]
    assert len(traces) == 4 and traces[0]["decisions"]
    l2 = [t for d in traces[0]["decisions"] for t in d["llm"] if t.get("role") == "L2" and "fallback" not in t]
    assert l2 and all(t["error"] for t in l2)
    ab = e1.run_AB(reduced=True, seeds=(10,), workers=1, out=str(tmp_path / "e1_AB.json"))
    an = e1.analyse(ab, cd)
    assert an["comparisons"]["arm_C-arm_B"]["status"] == "not_computed"
    assert an["success_criterion_met"] is None
    assert an["meta"]["reduced"] is True and an["meta"]["frozen"] == {"AB": False, "CD": False}
    # ledger: not held-out complete, not frozen, fallback-only -> Partial with reasons
    root = tmp_path / "root"
    (root / "results").mkdir(parents=True)
    (root / "results" / "e1_CD.json").write_text(out.read_text())
    (root / "results" / "e5_live.json").write_text(json.dumps({"meta": {}, "rows": []}))
    row = evidence_ledger.ledger(root)[5]
    assert row["status"] == "Partial"
    assert "invalid_mostly_fallback" in row["evidence"] and "held-out" in row["evidence"]
    assert "frozen" in row["evidence"]


def test_e1_cd_stops_on_account_error(no_frozen, tmp_path):
    """Balance runs out mid-run (HTTP 402): remaining jobs are skipped and the file says aborted."""
    from campusflex.llm.client import LLMResponse, ScriptedClient

    broke = ScriptedClient(lambda *a: LLMResponse("L2", None, error="api_status_402: Insufficient Balance"),
                           model="deepseek-flash")
    out = tmp_path / "e1_CD.json"
    cd = e1.run_CD(lambda: broke, reduced=True, seeds=(10,), threads=1, out=str(out))
    assert cd["status"] == "aborted" and "rows" not in cd and len(cd["rows_partial"]) == 1
    assert cd["meta"]["n_jobs"] == 4 and cd["meta"]["n_jobs_skipped"] == 3
    assert "Insufficient Balance" in cd["reason"] and "rerun" in cd["reason"]
    an = e1.analyse({"meta": {}, "rows": []}, cd)
    assert an["meta"]["CD"]["status"] == "aborted"
    root = tmp_path / "root"
    (root / "results").mkdir(parents=True)
    (root / "results" / "e1_CD.json").write_text(out.read_text())
    (root / "results" / "e5_live.json").write_text(json.dumps({"status": "not_run", "reason": "x"}))
    row = evidence_ledger.ledger(root)[5]
    assert row["status"] == "Not run" and "status aborted (LLM account error" in row["evidence"]


def test_ledger_e1_ab_executed_only_for_full_frozen_heldout(tmp_path):
    root = tmp_path
    (root / "results").mkdir()
    rows = [_e1_row(a, s, k, 1.0, "h") | {"mismatch": mm}
            for a in ("arm_A", "arm_B") for s in ("heat", "fault", "fault_surge", "double_fault")
            for mm in (False, True) for k in HELDOUT_SEEDS]
    meta = {"seeds": list(HELDOUT_SEEDS), "frozen": True, "config_hash": "h", "reduced": False}
    (root / "results" / "e1_AB.json").write_text(json.dumps({"meta": meta, "rows": rows}))
    assert evidence_ledger.ledger(root)[4]["status"] == "Executed"
    (root / "results" / "e1_AB.json").write_text(json.dumps({"meta": {**meta, "frozen": False}, "rows": rows}))
    assert evidence_ledger.ledger(root)[4]["status"] == "Partial"
    (root / "results" / "e1_AB.json").write_text(json.dumps({"meta": meta, "rows": rows[:-1]}))
    r = evidence_ledger.ledger(root)[4]
    assert r["status"] == "Partial" and "159 episode rows" in r["evidence"]


# ------------------------------------------------------------------ E2 / E3 / E4
def test_e2_dead_llm_arm_is_invalid_without_summary(tmp_path):
    res = e2.run(dead_client(), tmp_path / "e2.json", direction=False)
    arm = res["arms"]["llm_compiler"]
    assert arm["status"] == "invalid_mostly_fallback" and "summary" not in arm
    assert arm["llm"]["n_calls"] == 40 and arm["llm"]["n_failed_calls"] == 40
    res = e2.run(None, tmp_path / "e2.json", direction=False, llm_reason="preflight failed: x")
    assert res["arms"]["llm_compiler"] == {"status": "not_run", "reason": "preflight failed: x"}


def test_e3_paired_averages_repeats():
    rows = []
    for sd in range(10, 20):
        rows.append({"arm": "constant", "scenario": "s", "seed": sd, "repeat": 0, "score": 10.0})
        for rep, v in enumerate((10.0, 5.0, 5.0)):
            rows.append({"arm": "llm", "scenario": "s", "seed": sd, "repeat": rep, "score": v})
    p = e3.paired(rows, "llm", "constant", "score")
    assert p["n"] == 10 and p["mean_diff"] == pytest.approx(20 / 3 - 10)


def test_e3_defaults_and_dead_llm(no_frozen, tmp_path):
    import inspect

    assert inspect.signature(e3.run).parameters["repeats"].default == 3
    res = e3.run(dead_client(), tmp_path / "e3.json", seeds=(10,), scenarios=("notice_surge",), repeats=1, workers=1)
    assert res["status"]["llm"] == "invalid_mostly_fallback"
    assert res["gap_closed"]["status"] == "invalid_mostly_fallback" and "moves" not in res["gap_closed"]
    assert "llm" not in res["overall"] and not any(k.startswith("llm_") for k in res["paired"])
    assert res["llm_stats"]["fallback_rate"] == 1.0
    assert res["meta"]["frozen"] is False
    root = tmp_path / "root"
    (root / "results").mkdir(parents=True)
    (root / "results" / "e3.json").write_text((tmp_path / "e3.json").read_text())
    for f in ("e2", "e4"):
        (root / "results" / f"{f}.json").write_text(json.dumps({"arms": {"x": {"status": "run"}}}))
    row = evidence_ledger.ledger(root)[7]
    assert row["status"] == "Partial" and "e3.json" in row["evidence"]


def test_e4_no_success_without_texts():
    assert e4.evaluate([])["success_100pct_matched_or_flagged"] is None


# ------------------------------------------------------------------ E6
def _fake_sweep(cfg, seeds, scenarios, workers=None, policy="arm_B", **kw):
    """pre-review (default λΩ): U=10, score 50; L5 (λΩ=8): U=0, score 60; random (λΩ=2): U=10, score 49."""
    u, j = {PlannerConfig().lambda_Omega: (10, 50.0), 8.0: (0, 60.0), 2.0: (10, 49.0)}[cfg.lambda_Omega]
    return [{"scenario": sc, "seed": s, "mismatch": False, "unserved_person_min": u, "score": j + 0.01 * s,
             "kwh": 1.0, "dev_c": 0.0, "moves": 0, "switches": 0} for sc in scenarios for s in seeds]


@pytest.fixture
def e6_paths(tmp_path, monkeypatch, no_frozen):
    R = tmp_path / "results"
    R.mkdir(exist_ok=True)
    monkeypatch.setattr(e6, "RESULTS", R)
    monkeypatch.setattr(e6, "PATHS", {
        k: {"config": R / f"{k}_config.json", "history": R / f"{k}_history.json",
            "changelog": R / f"{k}_changelog.json", "report": R / f"{k}_report.md"} for k in ("llm", "random")})
    monkeypatch.setattr(e6, "sweep", _fake_sweep)
    return R


def _write_review(kind, lam, status="run", base=None):
    p = e6.PATHS[kind]
    replace(PlannerConfig(), lambda_Omega=lam).save(p["config"])
    p["history"].write_text(json.dumps({"status": status, "meta": {"protocol": "full",
                                                                   "base_config_hash": base or common.config_hash(PlannerConfig())},
                                        "accepted": [], "proposals": 3}))


def test_e6_eq4_order_and_invalid_l5(e6_paths):
    _write_review("llm", 8.0)
    _write_review("random", 2.0)
    out = e6.evaluate(quick=True)
    c = out["comparisons"]["L5_minus_pre_review"]
    assert c["decided_by"] == "unserved" and c["a_better"] and c["mean_diff"] > 0   # worse score, fewer unserved
    assert out["verdict"]["L5_beats_pre_review"] is True and out["verdict"]["success"] is True
    assert out["meta"]["frozen"] is False
    _write_review("llm", 8.0, status="invalid_mostly_fallback")
    out = e6.evaluate(quick=True)
    assert out["arms"]["L5"]["status"] == "invalid_mostly_fallback"
    assert out["comparisons"]["L5_minus_pre_review"] is None and "not evaluable" in out["verdict"]["result"]
    _write_review("llm", 8.0, base="other")
    assert e6.evaluate(quick=True)["arms"]["L5"]["status"] == "invalid_base_config_mismatch"


def test_e6_review_with_dead_llm_is_invalid(e6_paths):
    res = e6.run_review("llm", client=dead_client(), quick=True, workers=1)
    assert res["status"] == "invalid_mostly_fallback" and res["llm"]["n_failed_calls"] == res["llm"]["n_calls"] > 0
    assert json.loads(e6.PATHS["llm"]["history"].read_text())["status"] == "invalid_mostly_fallback"
    assert e6.run_review("llm", client=None, reason="preflight failed: x") == {"status": "not_run",
                                                                              "reason": "preflight failed: x"}


# ------------------------------------------------------------------ ledger + tables
def test_ledger_mqtt_skipped_is_partial(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_iot.py").write_text("")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "test_results.txt").write_text(
        "tests/test_iot.py::test_topic_wildcards PASSED\n"
        "tests/test_iot.py::test_full_episode_through_mqtt_broker_matches_direct SKIPPED\n")
    r = evidence_ledger.ledger(tmp_path)[2]
    assert r["status"] == "Partial" and "SKIPPED" in r["evidence"]
    (tmp_path / "results" / "test_results.txt").write_text(
        "tests/test_iot.py::test_topic_wildcards PASSED\n"
        "tests/test_iot.py::test_mqtt_unreachable_falls_back_and_reports_it PASSED\n"
        "tests/test_iot.py::test_full_episode_through_mqtt_broker_matches_direct[planner] PASSED\n"
        "tests/test_iot.py::test_runs_sharing_a_real_broker_do_not_control_each_other PASSED\n")
    r = evidence_ledger.ledger(tmp_path)[2]
    assert r["status"] == "Executed" and "incl. 2 test(s) through a real MQTT broker" in r["evidence"]


def test_tables_vgap_e1_e3_sections(tmp_path):
    with (tmp_path / "vgap_dev.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario", "seed", "mismatch", "policy", "lambda_Omega", "step",
                                          "pred_first_V", "real_first_V", "pred_V_horizon", "real_V_horizon"])
        w.writeheader()
        for lam in (0.0, 4.0):
            for mm in (False, True):
                w.writerow({"scenario": "fault", "seed": 0, "mismatch": mm, "policy": "B", "lambda_Omega": lam,
                            "step": 0, "pred_first_V": 1.0, "real_first_V": 1.5, "pred_V_horizon": 2.0,
                            "real_V_horizon": 3.0})
    (tmp_path / "vgap_dev_summary.json").write_text(json.dumps(
        {"omega_ablation_score_diff_lambda0_minus_declared": {"n": 80, "mean_diff": 14.9, "lo": 2.5, "hi": 26.6,
                                                              "excludes_zero": True}}))
    ab = [_e1_row(a, "fault", s, 10.0, "aaa") for a in ("arm_A", "arm_B") for s in range(10, 13)]
    cd = [_e1_row("arm_C", "fault", s, 5.0, "bbb") for s in range(10, 13)]
    cd += [_e1_row("arm_D", "fault", s, 5.0, "aaa", fallbacks=9) for s in range(10, 13)]
    (tmp_path / "e1_AB.json").write_text(json.dumps({"meta": {"config_hash": "aaa", "frozen": False}, "rows": ab}))
    (tmp_path / "e1_CD.json").write_text(json.dumps({"meta": {"config_hash": "bbb"}, "rows": cd}))
    (tmp_path / "e3.json").write_text(json.dumps({
        "status": {"constant": "run", "llm": "not_run"}, "status_reasons": {"llm": "preflight failed"},
        "overall": {"constant": {"n": 1, "score": 3.0}},
        "paired": {"keyword_vs_constant/score": {"n": 40, "mean_diff": -12.4, "lo": -19.2, "hi": -6.5,
                                                 "excludes_zero": True}},
        "gap_closed": {"status": "not_run"}, "rows": []}))
    text = tables.build_tables(tmp_path)
    vg = text[text.index("## Ledger item 1"):text.index("## E1")]
    assert "| fault | True | 4.0 | B | 1 | 1.0000 | 1.5000 | 0.5000 | 1.0000 |" in vg and "14.900" in vg
    e1s = text[text.index("## E1"):text.index("## E2")]
    assert "**Not frozen**" in e1s and "refused" in e1s and "config_hash differs" in e1s
    assert "D − B: not computed (D invalid_mostly_fallback)" in e1s
    assert "| fault | B − A |" in e1s and "C − B |" not in e1s
    e3s = text[text.index("## E3"):text.index("## E4")]
    assert "preflight failed" in e3s and "keyword_vs_constant/score" in e3s and "-12.400" in e3s
    assert "not computed: LLM arm not_run" in e3s


def test_e5_injection_pair_replays_identical_prompts():
    """A screened injection leaves the L3 prompt unchanged: the counterfactual run must reuse the
    factual reply, so a model that answers differently on a second identical call does not show up
    as injection-attributed divergence."""
    from campusflex.config import PlannerConfig
    from campusflex.experiments import e5
    from campusflex.experiments.e5_fixtures import load_all
    from campusflex.llm.client import ScriptedClient

    cfg = PlannerConfig()
    fx = load_all(live=True)[0]
    seen = []

    def flip_flop(role, system, user):           # different answer on every call
        seen.append(user)
        if len(seen) % 2:
            return {"adjustments": [{"source_id": "N1", "kind": "demand", "start": "20:30", "end": "22:00",
                                     "delta": 60, "confidence": "medium", "evidence": "expect about 60 students",
                                     "recheck_room": None}]}
        return {"adjustments": []}

    r = e5._injection_pair(fx, cfg, "direct", lambda: ScriptedClient(flip_flop), screen=True)
    assert r["nx_screened"] and len(seen) == 1 and r["divergence"] == 0
    seen.clear()
    r = e5._injection_pair(fx, cfg, "direct", lambda: ScriptedClient(flip_flop), screen=False)
    assert r["nx_sent_to_model"] and len(seen) == 2       # different prompts: both sent


def test_account_errors_abort_the_llm_arm_instead_of_counting_as_model_failures(tmp_path):
    """HTTP 401/402 (key, balance) says nothing about the model: E2/E4 mark the arm "aborted",
    the ledger does not count it, and llm_arm_stats ranks it above "invalid_mostly_fallback"."""
    from campusflex.config import PlannerConfig
    from campusflex.episode import run_episode
    from campusflex.experiments import e2, e4
    from campusflex.experiments.common import first_account_error, llm_arm_stats
    from campusflex.llm.client import LLMResponse, ScriptedClient
    from campusflex.policies import make_policy
    from campusflex.scenarios import make_scenario

    assert llm_arm_stats(10, 10, account_error="api_status_402: Insufficient Balance")["status"] == "aborted"
    assert llm_arm_stats(10, 10)["status"] == "invalid_mostly_fallback"
    assert first_account_error(["timeout", None, "api_status_401: bad key"]) == "api_status_401: bad key"
    assert first_account_error(["timeout", "api_status_500"]) is None

    broke = ScriptedClient(lambda role, s, u: LLMResponse(role, None, error="api_status_402: Insufficient Balance"))
    r2 = e2.run(broke, out=tmp_path / "e2.json", direction=False)
    arm = r2["arms"]["llm_compiler"]
    assert arm["status"] == "aborted" and "Insufficient Balance" in arm["reason"] and "summary" not in arm

    cfg = PlannerConfig()
    ep = tmp_path / "ep.json"
    run_episode(make_scenario("fault", 10), make_policy("arm_A", cfg), cfg, keep_decisions=True).save(ep)
    r4 = e4.run(broke, out=tmp_path / "e4.json", episodes=[str(ep)], per_episode=2)
    assert r4["arms"]["llm"]["status"] == "aborted" and r4["arms"]["template"]["status"] == "run"


def test_ledger_checks_freeze_record_cross_arm_hash_and_items_6_to_8(tmp_path):
    """Executed only when the artifact matches the current freeze record, C/D share B's config, and
    E3/E4 follow the held-out protocol."""
    from campusflex.config import HELDOUT_SEEDS
    from campusflex.experiments.e1 import expected_episodes

    res = tmp_path / "results"
    res.mkdir()
    (res / "frozen_config.json").write_text(json.dumps(
        {"config_hash": "aaa", "notices_hash": "nnn", "llm": {"model": "deepseek-flash", "prompt_hashes": {"L2": "p2"}}}))

    def e1(arms, h, n):
        return {"meta": {"seeds": list(HELDOUT_SEEDS), "frozen": True, "config_hash": h, "prompt_hashes": {"L2": "p2"},
                         "model": "deepseek-flash"},
                "rows": [{"arm": a, "config_hash": h, "llm_l2_calls": 1} for a in arms for _ in range(n)]}

    n_ab = expected_episodes("AB", False) // 2
    (res / "e1_AB.json").write_text(json.dumps(e1(["arm_A", "arm_B"], "aaa", n_ab)))
    rows = {r["#"]: r for r in evidence_ledger.ledger(tmp_path)}
    assert rows[4]["status"] == "Executed", rows[4]
    (res / "frozen_config.json").write_text(json.dumps({"config_hash": "bbb"}))     # re-frozen later
    rows = {r["#"]: r for r in evidence_ledger.ledger(tmp_path)}
    assert rows[4]["status"] == "Partial" and "differs from the freeze record" in rows[4]["evidence"]
    (res / "frozen_config.json").write_text(json.dumps({"config_hash": "aaa"}))
    n_cd = expected_episodes("CD", False) // 2
    (res / "e1_CD.json").write_text(json.dumps(e1(["arm_C", "arm_D"], "ccc", n_cd)))
    rows = {r["#"]: r for r in evidence_ledger.ledger(tmp_path)}
    assert rows[5]["status"] == "Partial" and "C/D cannot be compared with B" in rows[5]["evidence"]
    (res / "e3.json").write_text(json.dumps({"meta": {"seeds": [0, 1], "scenarios": ["notice_surge"]},
                                             "status": {"constant": "run"}}))
    rows = {r["#"]: r for r in evidence_ledger.ledger(tmp_path)}
    assert rows[7]["status"] == "Partial" and "held-out seeds" in rows[7]["evidence"]


def test_scenario_generator_is_part_of_the_freeze_record(tmp_path, monkeypatch):
    """Changing a scenario input after the freeze gate marks held-out results scenario_frozen=false
    and the ledger reports the drift."""
    from dataclasses import replace as dc_replace

    from campusflex import scenarios

    monkeypatch.setattr(common, "FROZEN", tmp_path / "results" / "frozen_config.json")
    freeze.freeze(tmp_path / "results" / "frozen_config.json")
    m = common.freeze_meta("x", True)
    assert m["scenario_frozen"] is True and m["scenario_hash"] == common.scenario_hash()
    monkeypatch.setattr(scenarios, "INPUTS", dc_replace(scenarios.INPUTS, tout_start_c=31.0))
    m2 = common.freeze_meta("x", True)
    assert m2["scenario_frozen"] is False and "differs from the freeze gate" in m2["scenario_note"]
    drift = evidence_ledger._freeze_drift(tmp_path, "results/e1_AB.json",
                                          {"meta": {"scenario_hash": m2["scenario_hash"]}, "rows": [{}]})
    assert any("scenario generator differs" in d for d in drift)
