"""Experiment E1 – staged plans (RQ2), the central test (Sections 4.4, Table 13).

Arms: A masks only (v1); B masks + rule-staged; C masks + LLM-staged; D masks + rule + LLM.
Scenarios: heat, fault, fault during surge, double fault; each with and without model
mismatch; held-out seeds 10–19. A and B are deterministic and run once (2 × 80 episodes);
C and D run three times with caching disabled (2 × 240). Comparisons are paired by seed and
condition with 10,000 bootstrap resamples, in the order of Equation 4 (unserved person-minutes
first, then the score). Success: C or D beats B with an interval that excludes zero (lower is
better).

Reduced protocol (``--reduced``): fault and fault-during-surge only, no mismatch, one
repeat – 2 × 10 × 4 = 80 episodes over the four arms; results are labelled as reduced.

Guards: every held-out run reads the frozen configuration (``python -m campusflex freeze``) and
tags each row with its ``config_hash``; arms run under different configurations are never
compared. An LLM arm whose L2 calls mostly fell back is marked ``invalid_mostly_fallback`` and
excluded from the comparisons. Per-episode decisions and LLM transcripts of C/D are saved
(``e1_CD_traces.jsonl.gz``) because the uncached repeats cannot be replayed from the cache.
"""
from __future__ import annotations

import argparse
import gzip
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..config import HELDOUT_SEEDS
from ..episode import run_episode
from ..llm.client import PROVIDER_MODELS
from ..policies import make_policy
from ..scenarios import E1_SCENARIOS, make_scenario
from ..stats import group_means
from .common import (
    ROOT,
    compare_arms,
    config_conflict,
    config_hashes,
    episode_job,
    freeze_meta,
    frozen_config,
    llm_client,
    llm_freeze_meta,
    llm_stats_by_arm,
    read_json,
    run_parallel,
    write_json,
)

METRICS = ("score", "dev_c", "moves", "kwh", "unserved_person_min", "staged_selection_rate", "llm_selection_rate",
           "llm_fallback_rate", "llm_valid_proposal_rate")
LLM_ARMS = ("arm_C", "arm_D")
TRACE_KEYS = ("step", "state_version", "plan_id", "source", "trigger", "ranking", "n_candidates", "n_rejected",
              "rejected_sample", "llm", "macro", "notes", "committed", "ack")


def _grid(reduced: bool) -> tuple[tuple[str, ...], tuple[bool, ...], int]:
    if reduced:
        return ("fault", "fault_surge"), (False,), 1
    return E1_SCENARIOS, (False, True), 3


def expected_episodes(arms: str, reduced: bool, n_seeds: int = len(HELDOUT_SEEDS)) -> int:
    """Episodes in one protocol run: AB = 2 arms × grid; CD = 2 arms × grid × repeats."""
    scen, mms, reps = _grid(reduced)
    per_arm = len(scen) * len(mms) * n_seeds
    return 2 * per_arm * (reps if arms == "CD" else 1)


def _protocol_meta(arms: str, reduced: bool, seeds, h: str, frozen: bool) -> dict[str, Any]:
    scen, mms, reps = _grid(reduced)
    return {"reduced": reduced, "protocol": "reduced" if reduced else "full", **freeze_meta(h, frozen),
            "seeds": list(seeds), "heldout_seeds": list(seeds) == list(HELDOUT_SEEDS),
            "scenarios": list(scen), "mismatch": list(mms),
            "expected_episodes": expected_episodes(arms, reduced, len(list(seeds)))}


def run_AB(reduced: bool = False, seeds=HELDOUT_SEEDS, workers: int | None = None,
           out: str = "results/e1_AB.json") -> dict[str, Any]:
    cfg, h, frozen = frozen_config()
    scen, mms, _ = _grid(reduced)
    t0 = time.time()
    specs = [{"scenario": s, "seed": k, "mismatch": mm, "policy": arm, "cfg": cfg.to_dict()}
             for arm in ("arm_A", "arm_B") for s in scen for mm in mms for k in seeds]
    rows = run_parallel(episode_job, specs, workers)
    for r in rows:
        r["arm"] = "arm_" + r["policy"]          # policy names are "A"/"B"
        r["repeat"] = 0
        r["config_hash"] = h
    res = {"meta": {"experiment": "E1 arms A,B", **_protocol_meta("AB", reduced, seeds, h, frozen),
                    "n_episodes": len(rows), "seconds": round(time.time() - t0, 1)},
           "rows": rows}
    write_json(out, res)
    return res


def _compact_decision(d: dict[str, Any]) -> dict[str, Any]:
    out = {k: d[k] for k in TRACE_KEYS if k in d}
    llm = []
    for t in d.get("llm") or []:
        t = dict(t)
        if t.get("data") is not None:
            t.pop("text", None)                  # the parsed reply is kept; the raw text duplicates it
        llm.append(t)
    out["llm"] = llm
    return out


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def run_CD(client_factory, reduced: bool = False, seeds=HELDOUT_SEEDS, threads: int = 8,
           out: str = "results/e1_CD.json", traces_out: str | Path | None = None) -> dict[str, Any]:
    """LLM arms. ``client_factory()`` must return an uncached, preflighted LLM client.

    An account error (HTTP 401/402, e.g. the balance ran out mid-run) stops the run: jobs not yet
    started are skipped and the file is written with ``status: aborted`` (rows kept only as
    ``rows_partial``), not as a fallback-only result."""
    from ..llm.client import ABANDONED, is_account_error
    from ..roles.l2_proposer import LLMStagedProposer

    cfg, h, frozen = frozen_config()
    scen, mms, repeats = _grid(reduced)
    jobs = [(arm, s, mm, k, rep) for arm in LLM_ARMS for s in scen for mm in mms for k in seeds
            for rep in range(repeats)]

    abort = threading.Event()
    account_errors: list[str] = []

    def one(job):
        arm, s, mm, k, rep = job
        if abort.is_set():
            return {"_skipped": True}, None
        try:
            client = client_factory()
            pol = make_policy(arm, cfg, l2=LLMStagedProposer(client, cfg))
            r = run_episode(make_scenario(s, k, mismatch=mm), pol, cfg, keep_decisions=True)
        except Exception as e:                   # one broken episode must not lose the whole run
            return {"_failed": True, "arm": arm, "scenario": s, "seed": k, "mismatch": mm, "repeat": rep,
                    "error": f"{type(e).__name__}: {e}"}, None
        calls = [t for d in r.decisions for t in d["llm"] if t.get("role") == "L2" and "fallback" not in t]
        acct = next((t["error"] for t in calls if is_account_error(t.get("error"))), None)
        if acct:
            account_errors.append(acct)
            abort.set()
        ps = r.policy_stats
        row = r.summary()
        row.update({
            "arm": arm, "repeat": rep, "config_hash": h,
            "llm_n_calls": int(ps.get("l2_calls", 0)),
            "llm_n_failed_calls": sum(1 for t in calls if t.get("error")),
            "llm_valid_proposal_rate": (ps["l2_valid"] / ps["l2_plans"]) if ps.get("l2_plans") else None,
            "llm_fallback_rate": (ps["l2_fallbacks"] / ps["l2_calls"]) if ps.get("l2_calls") else None,
            "llm_empty_rate": (ps.get("l2_empty", 0) / ps["l2_calls"]) if ps.get("l2_calls") else None,
            "llm_tokens_in": sum(t.get("input_tokens", 0) or 0 for t in calls),
            "llm_tokens_out": sum(t.get("output_tokens", 0) or 0 for t in calls),
            "llm_latency_s": round(sum(t.get("latency_s", 0.0) or 0.0 for t in calls), 2),
            "llm_models": sorted({t.get("model") for t in calls if t.get("model") and not t.get("error")}),
            "llm_cached_calls": sum(1 for t in calls if t.get("cached")),
        })
        trace = {"arm": arm, "scenario": s, "seed": k, "mismatch": mm, "repeat": rep, "config_hash": h,
                 "decisions": [_compact_decision(d) for d in r.decisions]}
        return row, trace

    t0 = time.time()
    abandoned0 = ABANDONED.snapshot()
    with ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
        results = list(ex.map(one, jobs))
    rows = [r for r, _ in results if not r.get("_failed") and not r.get("_skipped")]
    failed = [r for r, _ in results if r.get("_failed")]
    skipped = sum(1 for r, _ in results if r.get("_skipped"))
    tp = _resolve(traces_out) if traces_out else _resolve(out).with_name(Path(out).stem + "_traces.jsonl.gz")
    tp.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(tp, "wt", encoding="utf-8") as f:
        for _, tr in results:
            if tr is not None:
                f.write(json.dumps(tr, default=str, separators=(",", ":")) + "\n")
    stats = llm_stats_by_arm(rows)
    # the freeze gate compares the requested model id; the provider may report another name for it
    # (DeepSeek served the legacy id "deepseek-chat" as "deepseek-flash"), recorded as served_models
    served = sorted({m for r in rows for m in r.get("llm_models") or []})
    res = {"meta": {"experiment": "E1 arms C,D", **_protocol_meta("CD", reduced, seeds, h, frozen),
                    **llm_freeze_meta(model=getattr(client_factory(), "model", None)), "served_models": served,
                    "repeats": repeats, "n_episodes": len(rows), "failed_jobs": failed,
                    "caching": "disabled", "traces": str(tp), "llm_arm_stats": stats,
                    # calls given up at the step deadline: billed, but not in the rows' token counts
                    # (those that completed by the time the run ended)
                    "abandoned_calls": ABANDONED.delta(ABANDONED.snapshot(), abandoned0),
                    "seconds": round(time.time() - t0, 1)},
           "rows": rows}
    if abort.is_set():
        n_ok = sum(int(r["llm_n_calls"]) - int(r["llm_n_failed_calls"]) for r in rows)
        res["meta"].update({"n_jobs": len(jobs), "n_jobs_skipped": skipped, "account_error": account_errors[0]})
        res = {"meta": res["meta"], "status": "aborted",
               "reason": (f"LLM account error after {len(rows)} of {len(jobs)} episodes ({n_ok} L2 replies): "
                          f"{account_errors[0][:160]}. Fix the account (top up / key) and rerun."),
               "rows_partial": rows}
    write_json(out, res)
    return res


def _file_hash_defaults(ab: dict[str, Any], cd: dict[str, Any] | None) -> dict[str, str | None]:
    """File-level config hash per arm, for rows written before rows were tagged."""
    ha = (ab.get("meta") or {}).get("config_hash")
    hc = ((cd or {}).get("meta") or {}).get("config_hash")
    return {"arm_A": ha, "arm_B": ha, "arm_C": hc, "arm_D": hc}


def analyse(ab: dict[str, Any], cd: dict[str, Any] | None = None) -> dict[str, Any]:
    if cd is not None and (cd.get("status") == "not_run" or not cd.get("rows")):
        cd_meta = {**(cd.get("meta") or {}), "status": cd.get("status", "no rows"), "reason": cd.get("reason")}
        cd = None
    else:
        cd_meta = (cd or {}).get("meta")
    ab_meta = ab.get("meta") or {}
    cd_rows = list(cd["rows"]) if cd else []
    rows = list(ab.get("rows") or []) + cd_rows
    hashes = config_hashes(rows, "arm", _file_hash_defaults(ab, cd))
    llm_stats = llm_stats_by_arm(cd_rows)
    per_arm = group_means(rows, ("arm",), METRICS)
    for r in per_arm:
        if r["arm"] in llm_stats:
            r["llm"] = llm_stats[r["arm"]]
            r["status"] = llm_stats[r["arm"]]["status"]
        else:
            r["status"] = "run"
    out: dict[str, Any] = {
        "meta": {"AB": ab_meta, "CD": cd_meta,
                 "reduced": bool(ab_meta.get("reduced")) or bool((cd_meta or {}).get("reduced")),
                 "frozen": {"AB": ab_meta.get("frozen"), "CD": (cd_meta or {}).get("frozen")},
                 "config_hash": hashes, "order": "Eq. 4: unserved person-minutes first, then score"},
        "per_arm": per_arm,
        "per_scenario": group_means(rows, ("scenario", "mismatch", "arm"), METRICS),
        "llm_arm_stats": llm_stats,
        "comparisons": {},
    }
    pairs = [("arm_B", "arm_A")]
    if cd:
        pairs += [("arm_C", "arm_B"), ("arm_D", "arm_B"), ("arm_C", "arm_A")]
    for x, y in pairs:
        name = f"{x}-{y}"
        bad = [a for a in (x, y) if a in llm_stats and llm_stats[a]["status"] != "run"]
        if bad:
            out["comparisons"][name] = {"status": "not_computed",
                                        "reason": "; ".join(f"{a}: {llm_stats[a]['status']} – {llm_stats[a].get('reason')}"
                                                            for a in bad)}
            continue
        conflict = config_conflict(hashes, x, y)
        if conflict:
            out["comparisons"][name] = {"status": "refused_config_mismatch", "reason": conflict}
            continue
        overall = compare_arms(rows, "arm", x, y)
        if overall is None:
            continue
        per = {}
        for s in sorted({r["scenario"] for r in rows}):
            c = compare_arms([r for r in rows if r["scenario"] == s], "arm", x, y)
            if c is not None:
                per[s] = {k: v for k, v in c.items() if k != "keys"}
        out["comparisons"][name] = {"status": "computed", "overall": {k: v for k, v in overall.items() if k != "keys"},
                                    "per_scenario": per}
    if cd:
        done = {c: out["comparisons"][c] for c in ("arm_C-arm_B", "arm_D-arm_B")
                if out["comparisons"].get(c, {}).get("status") == "computed"}
        win = [c for c, v in done.items() if v["overall"]["a_better"]]
        out["success_criterion_met"] = bool(win) if done else None
        out["winning_arms"] = win
        if not done:
            out["success_note"] = "no valid C/D comparison (see comparisons[*].reason)"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="E1 staged-plan experiment")
    ap.add_argument("--arms", default="AB", choices=["AB", "CD", "all"])
    ap.add_argument("--reduced", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--provider", default=None, choices=list(PROVIDER_MODELS),
                    help="LLM provider for arms C/D (default: CAMPUSFLEX_PROVIDER or deepseek)")
    ap.add_argument("--model", default=None, help="model id (default: provider default)")
    a = ap.parse_args()
    if a.arms in ("AB", "all"):
        ab = run_AB(a.reduced, workers=a.workers)
    else:
        p = ROOT / "results" / "e1_AB.json"
        if not p.exists():
            raise SystemExit("results/e1_AB.json missing: run `python -m campusflex e1 --arms AB` first")
        ab = read_json(p)
    cd = None
    if a.arms in ("CD", "all"):
        from ..llm.client import make_client

        client, reason = llm_client(cache=False, provider=a.provider, model=a.model)
        if client is None:
            write_json("results/e1_CD.json", {"meta": {"experiment": "E1 arms C,D", "reduced": a.reduced},
                                               "status": "not_run", "reason": reason})
            print("LLM arms not run:", reason)
        else:
            cd = run_CD(lambda: make_client(a.provider, cache=False, model=a.model), a.reduced, threads=a.threads)
            if cd.get("status") == "aborted":
                print("LLM arms aborted:", cd["reason"])
    an = analyse(ab, cd)
    write_json("results/e1_analysis.json", an)
    print(json.dumps({"per_arm": an["per_arm"],
                      "comparisons": {k: v.get("overall", v) for k, v in an["comparisons"].items()}},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
