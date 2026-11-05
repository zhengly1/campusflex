"""L5 reviewer, random-search control, acceptance rule, tables and evidence ledger."""
from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from campusflex.config import L5_PARAM_SPACE, PlannerConfig, clock_str
from campusflex.experiments import evidence_ledger, tables
from campusflex.roles.l5_reviewer import (
    DIGEST_TOKEN_BUDGET,
    PROPOSAL_SCHEMA,
    ProposalResult,
    RandomSearchProposer,
    ReviewLoop,
    acceptance_rule,
    apply_diff,
    build_digest,
    check_proposal,
    space_index,
    sweep,
    write_report,
)

OMEGA_BONUS = {0.0: 0.0, 2.0: -1.0, 4.0: -2.0, 8.0: -3.0}


def fake_steps(seed: int) -> list[dict]:
    return [
        {"clock": clock_str(18 * 60 + 10 * k), "E": 1.0 + 0.01 * seed, "V": 0.02 * k, "M": k % 3, "K": int(k == 0),
         "pred_first_V": 0.01 * k, "source": "mask" if k % 2 else "rule", "unserved": 0}
        for k in range(24)
    ]


def fake_sweep(cfg, seeds, scenarios, keep):
    rows = []
    for sc in scenarios:
        for s in seeds:
            rows.append({
                "scenario": sc, "seed": s, "mismatch": False,
                "score": 50.0 + 0.3 * s + OMEGA_BONUS[float(cfg.lambda_Omega)],
                "kwh": 30.0, "V_deg_person_h": 1.0, "dev_c": 0.01, "moves": 2, "switches": 4,
                "unserved_person_min": 0, "hard_violations": 0,
                "_steps": fake_steps(s) if keep else None,
            })
    return rows


@pytest.fixture
def digest():
    cfg = PlannerConfig()
    return build_digest(fake_sweep(cfg, (0, 1, 2), ("fault", "heat"), True), cfg, [])


def good(cfg: PlannerConfig, digest: dict, **diff) -> dict:
    d = {"param": "rollout_horizon_min", "from": cfg.rollout_horizon_min, "to": 80}
    d.update(diff)
    return {
        "review_id": "R1-p1",
        "evidence": [{"interval": digest["interval_ids"][0], "observation": "predicted V far below realized V"}],
        "diagnosis": "horizon misses warming after the fault",
        "diff": d,
        "expected_effect": {"metric": "V", "direction": "decrease", "scenarios": ["fault"]},
        "risk": "earlier pre-cooling may raise energy",
    }


# ------------------------------------------------------------------ evidence checker
def test_valid_proposal_passes(digest):
    cfg = PlannerConfig()
    assert check_proposal(good(cfg, digest), digest, cfg, []) == []


@pytest.mark.parametrize(
    "mutate,code",
    [
        (lambda p: p["diff"].update({"to": 100}), "out_of_range"),
        (lambda p: p["diff"].update({"param": "band_high_c", "from": 27.0, "to": 28.0}), "forbidden_param"),
        (lambda p: p["diff"].update({"param": "max_moves_per_student", "from": 2, "to": 3}), "forbidden_param"),
        (lambda p: p["diff"].update({"from": 40, "to": 60}), "wrong_from"),
        (lambda p: p["diff"].update({"param": "lambda_Omega", "from": 4.0, "to": 0.0}), "not_one_step"),
        (lambda p: p["evidence"][0].update({"observation": "V was 0.4 above prediction"}), "number_in_observation"),
        (lambda p: p["evidence"][0].update({"interval": "fault/s99/20:00-20:40"}), "unknown_interval"),
        (lambda p: p.update({"evidence": []}), "no_evidence"),
    ],
)
def test_checker_rejects(digest, mutate, code):
    cfg = PlannerConfig()
    p = good(cfg, digest)
    mutate(p)
    codes = {i.code for i in check_proposal(p, digest, cfg, [])}
    assert code in codes


def test_checker_rejects_repeated_rejected_diff(digest):
    cfg = PlannerConfig()
    hist = [{"param": "rollout_horizon_min", "from": 60, "to": 80, "valid": True, "accepted": False}]
    codes = {i.code for i in check_proposal(good(cfg, digest), digest, cfg, hist)}
    assert "repeat_rejected" in codes


def test_precool_set_list_values(digest):
    cfg = PlannerConfig()
    p = good(cfg, digest, param="precool_set_min", **{"from": [10, 20, 30], "to": [20, 30, 40]})
    assert check_proposal(p, digest, cfg, []) == []
    assert apply_diff(cfg, "precool_set_min", [20, 30, 40]).precool_set_min == [20, 30, 40]


def test_schema_param_enum_matches_space():
    assert PROPOSAL_SCHEMA["properties"]["diff"]["properties"]["param"]["enum"] == list(L5_PARAM_SPACE)


# ------------------------------------------------------------------ random search
def test_random_proposer_emits_valid_one_step_diffs(digest):
    rp = RandomSearchProposer(seed=3)
    for cfg in (PlannerConfig(), replace(PlannerConfig(), lambda_Omega=8.0, rollout_horizon_min=40, lambda_M=0.02)):
        for k in range(60):
            res = rp.propose(digest, cfg, [], f"R1-p{k}")
            assert res.proposal is not None
            assert check_proposal(res.proposal, digest, cfg, [], require_evidence=False) == []
            d = res.proposal["diff"]
            assert abs(space_index(d["param"], d["to"]) - space_index(d["param"], d["from"])) == 1


def test_random_proposer_skips_rejected(digest):
    cfg = PlannerConfig()
    rp = RandomSearchProposer(0)
    hist = []
    for p, space in L5_PARAM_SPACE.items():
        k = space_index(p, getattr(cfg, p))
        for j in (k - 1, k + 1):
            if 0 <= j < len(space):
                hist.append({"param": p, "from": getattr(cfg, p), "to": space[j], "valid": True, "accepted": False})
    assert rp.propose(digest, cfg, hist, "R1-p1").proposal is None


# ------------------------------------------------------------------ acceptance rule
def _rows(scores, unserved=0, viol=0, sc="fault"):
    return [{"scenario": sc, "seed": i, "mismatch": False, "score": s, "unserved_person_min": unserved,
             "hard_violations": viol} for i, s in enumerate(scores)]


def test_acceptance_rule():
    base = _rows([50, 51, 52, 53, 54, 55, 56])
    better = _rows([49, 50.1, 50.9, 52.2, 52.8, 54.1, 55.0])
    assert acceptance_rule(base, better, _rows([60, 61]), _rows([59, 60]))["accepted"]
    # worse on check seeds
    r = acceptance_rule(base, better, _rows([60, 61]), _rows([62, 61]))
    assert not r["accepted"] and r["stage"] == "confirmation"
    # mixed: interval includes zero
    mixed = _rows([49, 52, 51, 54, 53, 56, 55])
    assert not acceptance_rule(base, mixed, _rows([60]), _rows([59]))["accepted"]
    # unserved increases in a scenario
    r = acceptance_rule(base, _rows([45, 46, 47, 48, 49, 50, 51], unserved=10), _rows([60]), _rows([59]))
    assert not r["accepted"] and "unserved" in r["tuning"]["reason"]
    # hard violations increase
    r = acceptance_rule(base, _rows([45, 46, 47, 48, 49, 50, 51], viol=1), _rows([60]), _rows([59]))
    assert not r["accepted"]


# ------------------------------------------------------------------ digest
def test_digest_deterministic_and_bounded():
    cfg = PlannerConfig()
    rows = fake_sweep(cfg, (0, 1, 2, 3, 4, 5, 6), ("fault", "heat", "steady", "notice_surge"), True)
    hist = [{"review_id": f"R{i}-p1", "param": "lambda_T", "from": 8.0, "to": 16.0, "accepted": False,
             "stage": "tuning", "tuning": {"mean_diff": 0.1, "lo": -0.2, "hi": 0.4}} for i in range(30)]
    d1, d2 = build_digest(rows, cfg, hist), build_digest(list(reversed(rows)), cfg, hist)
    assert json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True)
    assert len(json.dumps(d1)) / 4 <= DIGEST_TOKEN_BUDGET
    assert len(d1["top_intervals"]) == 5
    for iv in d1["interval_ids"]:
        assert re.fullmatch(r"[a-z_]+/s\d+/\d\d:\d\d-\d\d:\d\d", iv)
    top = d1["top_intervals"][0]
    assert top["pred_V"] is not None and top["realized_V"] >= top["pred_V"]


# ------------------------------------------------------------------ loop + report
class ScriptedL5:
    name = "L5"
    require_evidence = True

    def propose(self, digest, cfg, history, review_id):
        space = L5_PARAM_SPACE["lambda_Omega"]
        k = space_index("lambda_Omega", cfg.lambda_Omega)
        to = space[min(k + 1, len(space) - 1)]
        return ProposalResult({
            "review_id": review_id,
            "evidence": [{"interval": digest["interval_ids"][0], "observation": "occupants left in a warming room"}],
            "diagnosis": "terminal term too weak",
            "diff": {"param": "lambda_Omega", "from": cfg.lambda_Omega, "to": to},
            "expected_effect": {"metric": "V", "direction": "decrease", "scenarios": ["fault"]},
            "risk": "more pre-cooling energy",
        })


def test_review_loop_accepts_commits_and_stops(tmp_path: Path):
    loop = ReviewLoop(ScriptedL5(), PlannerConfig(), scenarios=("fault", "heat"), tuning_seeds=(0, 1, 2, 3),
                      check_seeds=(7, 8), sweep_fn=fake_sweep, out_config=tmp_path / "cfg.json",
                      changelog=tmp_path / "log.json", n_boot=2000)
    out = loop.run()
    assert [h["to"] for h in out.accepted] == [8.0]
    assert PlannerConfig.load(tmp_path / "cfg.json").lambda_Omega == 8.0
    log = json.loads((tmp_path / "log.json").read_text())
    assert log[0]["review_id"] == "R1-p1" and "R1-p1" in log[0]["commit_message"]
    assert out.proposals <= 18 and out.rounds <= 6
    # after acceptance, 8→8 proposals are invalid; two empty rounds stop the loop early
    assert out.rounds == 3
    rep = write_report(out, tmp_path / "l5_report.md")
    assert rep["faithfulness"]["flagged"] == []
    assert "lambda_Omega" in (tmp_path / "l5_report.md").read_text()


def test_report_numbers_verified_against_sweep_values_only(tmp_path: Path):
    from campusflex.roles.l5_reviewer import ReviewOutcome

    h = {"review_id": "R1-p1", "param": "rollout_horizon_min", "from": 60, "to": 80, "accepted": True, "valid": True,
         "stage": "accepted",
         "proposal": {"evidence": [{"interval": "fault/s4/20:00-20:40", "observation": "late warming"}],
                      "diagnosis": "the 80-min horizon cuts unserved minutes by 412 and saves 37 kWh"},
         "transcript": {"input_tokens": 3712, "output_tokens": 412, "latency_s": 37.2, "data": {"to": 80}},
         "tuning": {"mean_diff": -0.81, "lo": -1.2, "hi": -0.3, "n": 21},
         "confirmation": {"base_mean": 51.2, "cand_mean": 50.9}}
    out = ReviewOutcome(label="L5", initial_cfg={}, final_cfg={}, history=[h], rounds=1, proposals=1)
    rep = write_report(out, tmp_path / "r.md")
    assert set(rep["faithfulness"]["flagged"]) == {"412", "37"}     # the proposer's own numbers never verify
    text = (tmp_path / "r.md").read_text()
    assert "412" not in text and "-0.810" in text and "51.200" in text


def test_random_review_loop_budget(tmp_path: Path):
    loop = ReviewLoop(RandomSearchProposer(1), PlannerConfig(), scenarios=("fault",), tuning_seeds=(0, 1, 2),
                      check_seeds=(7,), sweep_fn=fake_sweep, stop_after_empty_rounds=99, n_boot=1000)
    out = loop.run()
    assert out.proposals <= 18
    assert out.sweeps["tuning"] <= 18 and out.sweeps["confirmation"] <= 18
    assert all(h["valid"] for h in out.history if h.get("proposal"))


def test_real_sweep_feeds_digest():
    cfg = PlannerConfig()
    rows = sweep(cfg, (0,), ("steady",), keep_steps=True, workers=1)
    assert rows[0]["policy"] == "B" and len(rows[0]["_steps"]) == 24
    d = build_digest(rows, cfg, [])
    assert d["interval_ids"] and d["scenarios"][0]["scenario"] == "steady"


# ------------------------------------------------------------------ tables + ledger
def test_tables_on_empty_dir(tmp_path: Path):
    text = tables.build_tables(tmp_path)
    assert "not run (artifact missing)" in text
    assert "Table 12" in text


def test_tables_with_v1_rows(tmp_path: Path):
    rows = []
    for pol, kwh, mv in (("fixed", 52.0, 0), ("threshold", 51.7, 0), ("planner", 38.0, 12), ("planner_nomove", 37.6, 45)):
        for s in range(5):
            rows.append({"scenario": "fault", "seed": s, "mismatch": False, "policy": pol, "kwh": kwh + 0.1 * s,
                         "service_pct": 100.0, "dev_c": 0.01, "moves": mv, "switches": 4, "score": 90.0})
    (tmp_path / "v1").mkdir()
    (tmp_path / "v1" / "summary.json").write_text(json.dumps({"meta": {}, "rows": rows}))
    (tmp_path / "e5_scripted.json").write_text(json.dumps({"meta": {"protocol": "full"},
                                                           "summary": {"unvalidated_executed": 0},
                                                           "by_class": [{"class": "schema", "fixtures": 30}]}))
    text = tables.build_tables(tmp_path)
    assert "Simulation planner" in text and "planner vs Fixed opening" in text
    assert "unvalidated_executed" in text
    out = tables.main(["--results", str(tmp_path)])
    assert out.exists()


def test_evidence_ledger_empty_root(tmp_path: Path):
    rows = evidence_ledger.ledger(tmp_path)
    assert len(rows) == 10
    assert {r["status"] for r in rows} <= {"Not run", "Not started"}
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "e1_AB.json").write_text(json.dumps({"meta": {"protocol": "reduced"}, "rows": []}))
    rows = evidence_ledger.ledger(tmp_path)
    assert rows[4]["status"] == "Partial"


def test_evidence_ledger_docs_templates_not_started():
    root = Path(__file__).resolve().parents[1]
    rows = evidence_ledger.ledger(root)
    assert rows[9]["status"] == "Not started"     # templates only – no fabricated records
