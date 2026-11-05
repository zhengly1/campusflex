"""Ledger item 1: predicted-vs-realized V log and the Ω term (Section 3.2).

At every decision step the log records the V the planner predicted for the committed
interval beside the V realized in that same interval, plus the horizon prediction beside
the realized V over the same window. Development seeds only. A small ablation compares
λΩ = 0 with the declared λΩ so the non-LLM fix gets the first chance to close the gap.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..config import DEV_SEEDS, PlannerConfig
from ..scenarios import E1_SCENARIOS
from ..stats import group_means, pair_rows, paired_bootstrap
from .common import ROOT, episode_job, run_parallel, write_json

FIELDS = ["scenario", "seed", "mismatch", "policy", "lambda_Omega", "step", "clock", "source", "plan_id",
          "pred_first_V", "real_first_V", "pred_V_horizon", "real_V_horizon", "pred_omega", "pred_E", "real_E"]


def _rows_from_steps(r: dict[str, Any], H: int) -> list[dict[str, Any]]:
    steps = r["_steps"]
    out = []
    for i, s in enumerate(steps):
        if s.get("pred_first_V") is None:
            continue
        window = steps[i : i + H]
        out.append({
            "scenario": r["scenario"], "seed": r["seed"], "mismatch": r["mismatch"], "policy": r["policy"],
            "lambda_Omega": r["lambda_Omega"], "step": s["step"], "clock": s["clock"], "source": s.get("source"),
            "plan_id": s.get("plan_id"), "pred_first_V": s["pred_first_V"], "real_first_V": round(s["V"], 6),
            "pred_V_horizon": s.get("pred_V"), "real_V_horizon": round(sum(w["V"] for w in window), 6),
            "pred_omega": s.get("pred_omega"), "pred_E": s.get("pred_E"), "real_E": round(s["E"], 6),
        })
    return out


def run(seeds=DEV_SEEDS, workers: int | None = None, out_csv: str = "results/vgap_dev.csv") -> dict[str, Any]:
    base = PlannerConfig()
    specs = []
    for lam in (0.0, base.lambda_Omega):
        cfg = replace(base, lambda_Omega=lam)
        for sc in E1_SCENARIOS:
            for mm in (False, True):
                for k in seeds:
                    specs.append({"scenario": sc, "seed": k, "mismatch": mm, "policy": "arm_B", "cfg": cfg.to_dict(),
                                  "keep_steps": True, "lambda_Omega": lam, "tag_keys": ["lambda_Omega"]})
    results = run_parallel(episode_job, specs, workers)
    rows = []
    for r in results:
        rows += _rows_from_steps(r, base.horizon_steps)
    p = ROOT / out_csv
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    summ = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    a, b, _ = pair_rows(summ, "lambda_Omega", 0.0, base.lambda_Omega, ("scenario", "seed", "mismatch"))
    gap = group_means(
        [{**x, "gap_first": x["real_first_V"] - x["pred_first_V"],
          "gap_horizon": x["real_V_horizon"] - (x["pred_V_horizon"] or 0.0)} for x in rows],
        ("scenario", "mismatch", "lambda_Omega"), ("gap_first", "gap_horizon"))
    res = {"meta": {"experiment": "V-gap log + Omega ablation (dev seeds)", "seeds": list(seeds), "policy": "arm_B",
                    "csv": out_csv, "n_rows": len(rows)},
           "omega_ablation_score_diff_lambda0_minus_declared": a and paired_bootstrap(a, b),
           "gap_by_scenario": gap,
           "rows": summ}
    write_json(Path(out_csv).with_name("vgap_dev_summary.json"), res)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Predicted-vs-realized V log on development seeds")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    res = run(workers=a.workers)
    print(json.dumps({k: res[k] for k in ("meta", "omega_ablation_score_diff_lambda0_minus_declared")}, indent=1))


if __name__ == "__main__":
    main()
