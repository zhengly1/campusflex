"""Ledger item 0: the v1 numerical experiment – 4 scenarios × 10 paired seeds × 4 policies
(fixed opening, demand threshold, simulation planner, planner without relocation cost),
v1 settings (single 30-minute mask hold, no terminal term, highest-id departures).

This project's own v1 implementation, built from Equations 1-5 and Tables 4, 6 and 8 (no
earlier v1 code was used). Its results are the v1 baseline of this repository; they are not
compared with the figures printed in the v2.1 document.
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any

from ..config import DEV_SEEDS, PlannerConfig
from ..scenarios import V1_SCENARIOS
from ..stats import group_means, pair_rows, paired_bootstrap
from .common import episode_job, run_parallel, write_json

POLICIES = ("fixed", "threshold", "planner", "planner_nomove")


def run(seeds=DEV_SEEDS, workers: int | None = None, out: str = "results/v1/summary.json") -> dict[str, Any]:
    cfg = PlannerConfig.v1()
    t0 = time.time()
    specs = [{"scenario": s, "seed": k, "policy": p, "cfg": cfg.to_dict()}
             for s in V1_SCENARIOS for k in seeds for p in POLICIES]
    rows = run_parallel(episode_job, specs, workers)
    by_policy = group_means(rows, ("policy",), ("kwh", "service_pct", "dev_c", "moves", "switches", "score"))
    a, b, _ = pair_rows(rows, "policy", "fixed", "planner", ("scenario", "seed"), metric="kwh")
    save_fixed = paired_bootstrap(a, b)
    a, b, _ = pair_rows(rows, "policy", "threshold", "planner", ("scenario", "seed"), metric="kwh")
    save_thr = paired_bootstrap(a, b)
    mv = {r["policy"]: r["moves"] for r in by_policy}
    reduction = 100.0 * (1 - mv["planner"] / mv["planner_nomove"]) if mv.get("planner_nomove") else None
    from dataclasses import asdict

    from .. import scenarios

    res = {
        "meta": {"experiment": "v1 core (this project's implementation)", "n_episodes": len(rows),
                 "seeds": list(seeds), "scenarios": list(V1_SCENARIOS), "policies": list(POLICIES),
                 "config": cfg.to_dict(), "scenario_inputs": asdict(scenarios.INPUTS),
                 "seconds": round(time.time() - t0, 1)},
        "table12": by_policy,
        "by_scenario": group_means(rows, ("scenario", "policy"), ("kwh", "dev_c", "moves", "switches", "score")),
        "energy_saving_vs_fixed_kwh": save_fixed,
        "energy_saving_vs_threshold_kwh": save_thr,
        "relocation_cost_move_reduction_pct": round(reduction, 2) if reduction is not None else None,
        "rows": rows,
    }
    write_json(out, res)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Reconstructed v1 experiment (160 episodes)")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    res = run(workers=a.workers)
    print(json.dumps({k: res[k] for k in ("table12", "energy_saving_vs_fixed_kwh", "energy_saving_vs_threshold_kwh",
                                          "relocation_cost_move_reduction_pct")}, indent=1))


if __name__ == "__main__":
    main()
