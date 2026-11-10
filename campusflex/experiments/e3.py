"""E3 (RQ4): does interpreting text notices reduce forecast error, unserved demand and
relocations?

Arms (all with the same masks-only simulation planner, arm-A style):
    constant  – v1 forecast: hold current demand and weather
    keyword   – regex notice parser (non-LLM control)
    llm       – L3 event interpreter (only with --llm; otherwise "not_run")
    oracle    – structured ground truth of each notice

Metrics: unserved person-minutes, relocations, forecast MAE of the 30-minute-ahead demand
forecast, plus energy/deviation/score. "Gap closed" = (keyword − LLM)/(keyword − oracle);
the success criterion is that the LLM closes at least half the gap (Table 13).

Protocol guards: held-out seeds run under the frozen configuration (``meta.frozen``); the LLM
arm needs a preflighted client, runs three uncached repeats per seed by default (Section 4.2)
and is marked "invalid_mostly_fallback" – with no gap_closed or paired comparison – when most
of its L3 calls failed. Paired intervals average the repeats per (scenario, seed) first.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config import HELDOUT_SEEDS, N_STEPS, PlannerConfig
from ..episode import run_episode
from ..forecast import ConstantForecaster, NoticeForecaster
from ..llm.client import LLMClient
from ..policies import SimulationPlanner
from ..roles.l3_events import KeywordParser, LLMInterpreter, OracleInterpreter
from ..scenarios import NOTICE_SCENARIOS, make_scenario
from ..stats import pair_rows, paired_bootstrap
from .common import (
    MAX_FALLBACK_RATE,
    config_hash,
    first_account_error,
    freeze_meta,
    frozen_config,
    llm_arm_stats,
    llm_freeze_meta,
    notices_freeze_meta,
    run_parallel,
    write_json,
)
from .e2 import add_llm_args, llm_client_or_reason

ARMS = ("constant", "keyword", "llm", "oracle")
LEAD_STEPS = 3          # 30 minutes ahead
LLM_REPEATS = 3         # uncached repeats per held-out seed (Section 4.2)
METRICS = ("unserved_person_min", "moves", "forecast_mae_30min", "kwh", "dev_c", "score")


def forecast_mae(decisions: list[dict[str, Any]], demand: np.ndarray, lead: int = LEAD_STEPS) -> float | None:
    errs = []
    for d in decisions:
        fc = (d.get("forecast") or {}).get("demand")
        t = d["step"]
        if fc is None or len(fc) <= lead or t + lead >= N_STEPS:
            continue
        errs.append(abs(float(fc[lead]) - float(demand[t + lead])))
    return float(np.mean(errs)) if errs else None


def _forecaster(arm: str, cfg: PlannerConfig, client: LLMClient | None = None) -> Any:
    if arm == "constant":
        return ConstantForecaster()
    if arm == "keyword":
        return NoticeForecaster(KeywordParser(), cfg)
    if arm == "oracle":
        return NoticeForecaster(OracleInterpreter(), cfg)
    if arm == "llm":
        if client is None:
            raise ValueError("llm arm needs a client")
        return NoticeForecaster(LLMInterpreter(client, cfg), cfg)
    raise KeyError(arm)


def run_one(arm: str, scenario: str, seed: int, cfg: PlannerConfig, client: LLMClient | None = None,
            repeat: int = 0) -> dict[str, Any]:
    sc = make_scenario(scenario, seed)
    fc = _forecaster(arm, cfg, client)
    pol = SimulationPlanner(cfg, forecaster=fc, name=f"E3:{arm}")
    r = run_episode(sc, pol, cfg, keep_decisions=True)
    row = {
        "arm": arm, "scenario": scenario, "seed": seed, "repeat": repeat, "mismatch": False,
        **{k: r.metrics[k] for k in ("unserved_person_min", "moves", "kwh", "dev_c", "score", "service_pct",
                                     "switches", "hard_violations")},
        "forecast_mae_30min": forecast_mae(r.decisions, sc.demand),
    }
    log = getattr(fc, "log", [])
    row["adjustments_applied"] = sum(1 for e in log if e.get("applied") is not None)
    row["adjustments_dropped"] = sum(1 for e in log if e.get("applied") is None)
    # where the dropped proposals came from: a proposal for a distractor or an occupant report is
    # rightly refused; a dropped proposal for a real notice is a miss
    source = {n.notice_id: ("distractor" if n.distractor else "injected" if n.injected else
                            "report" if n.kind == "report" else "notice") for n in sc.notices}
    by_source: dict[str, int] = {}
    for e in log:
        if e.get("applied") is None:
            k = source.get(str((e.get("raw") or {}).get("source_id")), "unknown")
            by_source[k] = by_source.get(k, 0) + 1
    row["dropped_by_source"] = by_source
    true_ids = {n.notice_id for n in sc.notices if n.truth is not None}
    applied_ids = {str(e["applied"].get("source_id")) for e in log if e.get("applied") is not None}
    row["true_notices"] = len(true_ids)
    row["true_notices_applied"] = len(true_ids & applied_ids)
    row["recheck_requests"] = len(getattr(fc, "recheck_requests", []) or [])
    interp = getattr(fc, "interpreter", None)
    if isinstance(interp, LLMInterpreter):
        row["llm_calls"] = len(interp.transcripts)
        row["llm_failed_calls"] = sum(1 for t in interp.transcripts if t.get("error"))
        row["llm_cached_calls"] = sum(1 for t in interp.transcripts if t.get("cached"))
        row["llm_errors"] = interp.errors[:10]
        row["llm_account_error"] = first_account_error(t.get("error") for t in interp.transcripts)
        row["llm_input_tokens"] = sum(t.get("input_tokens", 0) for t in interp.transcripts)
        row["llm_output_tokens"] = sum(t.get("output_tokens", 0) for t in interp.transcripts)
        row["llm_latency_s"] = round(sum(t.get("latency_s", 0.0) for t in interp.transcripts), 3)
        row["model"] = getattr(client, "model", "")
    return row


def _job(spec: dict[str, Any]) -> dict[str, Any]:
    return run_one(spec["arm"], spec["scenario"], spec["seed"], PlannerConfig.from_dict(spec["cfg"]))


def _mean(rows: list[dict[str, Any]], m: str) -> float | None:
    v = [r[m] for r in rows if r.get(m) is not None]
    return float(np.mean(v)) if v else None


def summarize(rows: list[dict[str, Any]], arms_run: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"overall": {}, "per_scenario": []}
    for arm in arms_run:
        ar = [r for r in rows if r["arm"] == arm]
        out["overall"][arm] = {m: _mean(ar, m) for m in METRICS}
        out["overall"][arm]["n"] = len(ar)
    for sc in sorted({r["scenario"] for r in rows}):
        for arm in arms_run:
            ar = [r for r in rows if r["arm"] == arm and r["scenario"] == sc]
            if ar:
                out["per_scenario"].append({"scenario": sc, "arm": arm, "n": len(ar), **{m: _mean(ar, m) for m in METRICS}})
    return out


def gap_closed(overall: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if "llm" not in overall:
        return {"status": "not_run"}
    g: dict[str, Any] = {}
    for m in ("unserved_person_min", "moves", "forecast_mae_30min", "score"):
        kw, llm, orc = overall["keyword"].get(m), overall["llm"].get(m), overall["oracle"].get(m)
        if None in (kw, llm, orc) or abs(kw - orc) < 1e-9:
            g[m] = {"keyword": kw, "llm": llm, "oracle": orc, "closed": None, "note": "no gap between keyword and oracle"}
        else:
            g[m] = {"keyword": kw, "llm": llm, "oracle": orc, "closed": (kw - llm) / (kw - orc)}
    closed = [v["closed"] for v in g.values() if v.get("closed") is not None]
    g["success_at_least_half"] = (min(closed) >= 0.5) if closed else None
    return g


def paired(rows: list[dict[str, Any]], a: str, b: str, metric: str) -> dict[str, Any] | None:
    """Paired bootstrap a − b over (scenario, seed); repeats are averaged per pair first."""
    usable = [r for r in rows if r.get(metric) is not None]
    xa, xb, keys = pair_rows(usable, "arm", a, b, pair_keys=("scenario", "seed"), metric=metric)
    if not keys:
        return None
    return paired_bootstrap(xa, xb)


def llm_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lr = [r for r in rows if r["arm"] == "llm"]
    applied = sum(int(r.get("adjustments_applied") or 0) for r in lr)
    dropped = sum(int(r.get("adjustments_dropped") or 0) for r in lr)
    st = llm_arm_stats(sum(int(r.get("llm_calls") or 0) for r in lr),
                       sum(int(r.get("llm_failed_calls") or 0) for r in lr),
                       n_proposals=applied + dropped, n_valid=applied,
                       account_error=next((r["llm_account_error"] for r in lr if r.get("llm_account_error")), None))
    st["episodes"] = len(lr)
    st["cached_calls"] = sum(int(r.get("llm_cached_calls") or 0) for r in lr)
    if all("true_notices" in r for r in lr) and lr:
        tn = sum(r["true_notices"] for r in lr)
        st["true_notice_coverage"] = (sum(r["true_notices_applied"] for r in lr) / tn) if tn else None
        drops: dict[str, int] = {}
        for r in lr:
            for k, v in (r.get("dropped_by_source") or {}).items():
                drops[k] = drops.get(k, 0) + v
        st["dropped_by_source"] = drops
        st["note"] = ("valid_proposal_rate counts every refused proposal, including proposals for distractors "
                      "and occupant reports that grounding rightly refuses; true_notice_coverage is the share "
                      "of real notices that got an applied adjustment")
    return st


def run(client: LLMClient | None = None, out: str | Path = "results/e3.json",
        seeds: tuple[int, ...] = HELDOUT_SEEDS, scenarios: tuple[str, ...] = NOTICE_SCENARIOS,
        arms: tuple[str, ...] = ARMS, repeats: int = LLM_REPEATS, workers: int | None = None,
        cfg: PlannerConfig | None = None, llm_reason: str | None = None) -> dict[str, Any]:
    fcfg, fh, ffrozen = frozen_config()
    cfg = cfg or fcfg
    h = config_hash(cfg)
    frozen = ffrozen and h == fh
    offline_arms = [a for a in arms if a != "llm"]
    specs = [{"arm": a, "scenario": s, "seed": sd, "cfg": cfg.to_dict()}
             for a in offline_arms for s in scenarios for sd in seeds]
    rows = run_parallel(_job, specs, workers)
    status = {a: "run" for a in offline_arms}
    reasons: dict[str, str] = {}
    stats = None
    if "llm" in arms:
        if client is None:
            status["llm"] = "not_run"
            reasons["llm"] = llm_reason or "no LLM client configured (use --llm)"
        else:
            for rep in range(repeats):
                for s in scenarios:
                    for sd in seeds:
                        rows.append(run_one("llm", s, sd, cfg, client, repeat=rep))
            stats = llm_stats(rows)
            status["llm"] = stats["status"]
            if stats["status"] != "run":
                reasons["llm"] = stats.get("reason", stats["status"])
    for r in rows:
        r["config_hash"] = h
    arms_run = [a for a in arms if status.get(a) == "run"]
    summ = summarize(rows, arms_run)
    result = {
        "experiment": "E3",
        "meta": {"experiment": "E3", **freeze_meta(h, frozen), **notices_freeze_meta(),
                 **(llm_freeze_meta(client) if client is not None else {}),
                 "seeds": list(seeds), "scenarios": list(scenarios),
                 "llm_repeats": repeats if client is not None else None,
                 "llm_caching": (None if client is None else
                                 ("enabled" if getattr(client, "enabled", False) else "disabled")),
                 "max_fallback_rate": MAX_FALLBACK_RATE},
        "seeds": list(seeds),
        "scenarios": list(scenarios),
        "status": status,
        "status_reasons": reasons,
        "planner": "masks only (arm A), constant weather",
        **summ,
        "gap_closed": gap_closed(summ["overall"]) if status.get("llm") == "run"
        else {"status": status.get("llm", "not_run"), **({"reason": reasons["llm"]} if "llm" in reasons else {})},
        "paired": {
            f"{a}_vs_constant/{m}": paired(rows, a, "constant", m)
            for a in arms_run if a != "constant" for m in ("unserved_person_min", "moves", "score")
        },
        "rows": rows,
    }
    if stats is not None:
        result["llm_stats"] = stats
    if status.get("llm") == "not_run":
        result["llm_note"] = f"LLM arm not run: {reasons['llm']}. No LLM result is reported."
    elif status.get("llm") not in (None, "run"):
        result["llm_note"] = f"LLM arm {status['llm']}: {reasons['llm']}. No LLM result is reported."
    write_json(out, result)
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_llm_args(p)
    p.add_argument("--out", default="results/e3.json")
    p.add_argument("--seeds", default="10-19", help="e.g. 10-19 or 0,1,2")
    p.add_argument("--scenarios", default=",".join(NOTICE_SCENARIOS))
    p.add_argument("--repeats", type=int, default=LLM_REPEATS, help="uncached LLM repeats per seed (default 3)")
    p.add_argument("--cache", action="store_true",
                   help="use the prompt-hash cache for the LLM arm (off by default: repeats measure model variance)")
    p.add_argument("--workers", type=int, default=None)
    a = p.parse_args(argv)
    if "-" in a.seeds:
        lo, hi = a.seeds.split("-")
        seeds = tuple(range(int(lo), int(hi) + 1))
    else:
        seeds = tuple(int(x) for x in a.seeds.split(","))
    client, reason = llm_client_or_reason(a, cache=a.cache and not a.no_cache)
    if client is None and a.llm:
        print("LLM arm not run:", reason)
    res = run(client, a.out, seeds, tuple(a.scenarios.split(",")), repeats=a.repeats, workers=a.workers,
              llm_reason=reason)
    print(json.dumps({"status": res["status"], "overall": res["overall"], "gap_closed": res["gap_closed"]},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
