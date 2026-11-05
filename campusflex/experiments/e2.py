"""E2 (RQ3): L1 intent compiler vs the structured-input form.

Metrics: field-level accuracy, hard-constraint omissions, weight-level direction correct,
clarification behaviour, and whether plans move in the intended direction when the compiled
weight levels are applied (planner on a development scenario, default vs compiled levels).

Success criterion (Table 13): zero omitted hard constraints; weight direction correct in
≥ 90 % of requests. The LLM arm runs only with ``--llm`` and after one real preflight call
(``common.llm_client``); otherwise it is reported as "not_run" – no result is ever fabricated.
An LLM arm whose calls mostly failed is reported as "invalid_mostly_fallback" without a summary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config import DEV_SEEDS, PlannerConfig
from ..llm.client import PROVIDER_MODELS, LLMClient
from ..roles.l1_intent import (
    WEIGHT_KEYS,
    IntentCompiler,
    IntentResult,
    request_fields,
    structured_form,
)
from .common import (
    ROOT,
    first_account_error,
    llm_arm_stats,
    llm_client,
    llm_freeze_meta,
    not_run,
    run_parallel,
    write_json,
)

DATA = ROOT / "data" / "e2_requests.json"
FIELDS = (
    "interval", "demand_estimate", "band_low_c", "band_high_c", "target_c", "energy_budget_kwh",
    "power_cap_kw", "max_moves_per_student", "no_move_windows", "lambda_M", "lambda_T", "lambda_S",
)
# Intended effect of each level on the planner (metric, sign of change vs default).
DIRECTION = {
    ("lambda_M", "high"): ("moves", -1), ("lambda_M", "low"): ("moves", +1),
    ("lambda_T", "high"): ("dev_c", -1), ("lambda_T", "low"): ("dev_c", +1),
    ("lambda_S", "high"): ("switches", -1), ("lambda_S", "low"): ("switches", +1),
}


def load_requests(path: Path = DATA) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text())["requests"]


def gold_fields(gold: dict[str, Any]) -> dict[str, Any]:
    """Gold labels in the same flat form as :func:`request_fields`."""
    f = {k: gold.get(k) for k in FIELDS if k not in WEIGHT_KEYS}
    f["max_moves_per_student"] = 2 if gold.get("max_moves_per_student") is None else gold["max_moves_per_student"]
    f["no_move_windows"] = sorted([list(w) for w in gold.get("no_move_windows") or []])
    for k in WEIGHT_KEYS:
        f[k] = (gold.get("weight_levels") or {}).get(k, "default")
    return f


def _eq(field: str, out: Any, gold: Any, out_all: dict[str, Any]) -> bool:
    if field == "no_move_windows" and out != gold:
        # Equivalent encoding: a window over the whole interval == zero moves allowed.
        iv = out_all.get("interval")
        if gold and iv and gold == [list(iv)] and out_all.get("max_moves_per_student") == 0:
            return True
    if isinstance(gold, (int, float)) and isinstance(out, (int, float)):
        return abs(float(gold) - float(out)) < 1e-6
    return out == gold


def score_request(item: dict[str, Any], res: IntentResult) -> dict[str, Any]:
    gold = gold_fields(item["gold"])
    out = request_fields(res.request)
    per_field = {f: (_eq(f, out.get(f), gold[f], out) if res.request is not None else False) for f in FIELDS}
    omitted = []
    for hc in item.get("hard_constraints", []):
        if hc == "band":
            ok = per_field["band_low_c"] and per_field["band_high_c"]
        else:
            ok = per_field.get(hc, False)
        if not ok:
            omitted.append(hc)
    levels_ok = all(per_field[k] for k in WEIGHT_KEYS)
    exp = item.get("expect_clarification")
    asked = bool(res.questions)
    return {
        "id": item["id"],
        "category": item["category"],
        "valid": res.request is not None,
        "field_correct": per_field,
        "field_accuracy": sum(per_field.values()) / len(FIELDS),
        "hard_omitted": omitted,
        "weights_correct": levels_ok,
        "weights_nondefault": any(gold[k] != "default" for k in WEIGHT_KEYS),
        "asked": asked,
        "n_questions": len(res.questions),
        "clarification_ok": None if exp is None else (asked == bool(exp)),
        "defaults_filled": res.defaults_filled,
        "levels": {k: out.get(k) for k in WEIGHT_KEYS},
        "errors": res.errors,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    nd = [r for r in rows if r["weights_nondefault"]]
    clar = [r for r in rows if r["clarification_ok"] is not None]
    by_cat: dict[str, list[float]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["field_accuracy"])
    return {
        "n": n,
        "valid_share": sum(r["valid"] for r in rows) / n,
        "field_accuracy": float(np.mean([r["field_accuracy"] for r in rows])),
        "field_accuracy_by_category": {k: float(np.mean(v)) for k, v in by_cat.items()},
        "hard_constraint_omissions": sum(len(r["hard_omitted"]) for r in rows),
        "weight_levels_correct_share": sum(r["weights_correct"] for r in rows) / n,
        "weight_direction_correct_share_nondefault": (sum(r["weights_correct"] for r in nd) / len(nd)) if nd else None,
        "clarification_correct_share": (sum(r["clarification_ok"] for r in clar) / len(clar)) if clar else None,
        "max_questions": max(r["n_questions"] for r in rows),
        "success": {
            "zero_hard_omissions": sum(len(r["hard_omitted"]) for r in rows) == 0,
            "weight_direction_ge_90pct": (sum(r["weights_correct"] for r in rows) / n) >= 0.9,
        },
    }


# ------------------------------------------------------------------ plan direction
def _direction_job(spec: dict[str, Any]) -> dict[str, Any]:
    from ..episode import run_episode
    from ..policies import SimulationPlanner
    from ..scenarios import make_scenario

    cfg = PlannerConfig.from_dict(spec["cfg"])
    sc = make_scenario(spec["scenario"], spec["seed"])
    r = run_episode(sc, SimulationPlanner(cfg, name="planner"), cfg)
    return {"levels": spec["levels"], "seed": spec["seed"], **{k: r.metrics[k] for k in ("moves", "dev_c", "switches", "kwh")}}


def plan_direction(level_sets: list[dict[str, str]], scenario: str = "fault", seeds: tuple[int, ...] = DEV_SEEDS[:3],
                   base: PlannerConfig | None = None, workers: int | None = None) -> dict[str, Any]:
    """Apply each distinct non-default level set and check the planner moves in the intended direction."""
    base = base or PlannerConfig()
    uniq: list[dict[str, str]] = [{k: "default" for k in WEIGHT_KEYS}]
    for ls in level_sets:
        ls = {k: ls.get(k, "default") for k in WEIGHT_KEYS}
        if ls not in uniq:
            uniq.append(ls)
    specs = [{"cfg": base.with_levels(ls).to_dict(), "levels": ls, "seed": s, "scenario": scenario}
             for ls in uniq for s in seeds]
    rows = run_parallel(_direction_job, specs, workers)
    def mean(ls: dict[str, str], m: str) -> float:
        return float(np.mean([r[m] for r in rows if r["levels"] == ls]))
    d0 = uniq[0]
    checks = []
    for ls in uniq[1:]:
        for k, v in ls.items():
            if v == "default":
                continue
            metric, sign = DIRECTION[(k, v)]
            delta = mean(ls, metric) - mean(d0, metric)
            checks.append({"levels": ls, "weight": k, "level": v, "metric": metric, "default_mean": mean(d0, metric),
                           "compiled_mean": mean(ls, metric), "delta": delta,
                           "direction": "intended" if np.sign(delta) == sign else ("unchanged" if delta == 0 else "opposite")})
    return {"scenario": scenario, "seeds": list(seeds), "checks": checks,
            "intended_share": (sum(c["direction"] == "intended" for c in checks) / len(checks)) if checks else None}


# --------------------------------------------------------------------------- run
def form_input(item: dict[str, Any]) -> dict[str, Any]:
    g = item["gold"]
    return {
        "interval": g.get("interval"),
        "demand_estimate": g.get("demand_estimate"),
        "band_low_c": g.get("band_low_c"),
        "band_high_c": g.get("band_high_c"),
        "target_c": g.get("target_c"),
        "energy_budget_kwh": g.get("energy_budget_kwh"),
        "power_cap_kw": g.get("power_cap_kw"),
        "max_moves_per_student": g.get("max_moves_per_student"),
        "no_move_windows": g.get("no_move_windows"),
        "weight_levels": g.get("weight_levels"),
    }


def run(client: LLMClient | None = None, out: str | Path = "results/e2.json", direction: bool = True,
        scenario: str = "fault", seeds: tuple[int, ...] = DEV_SEEDS[:3], workers: int | None = None,
        llm_reason: str | None = None) -> dict[str, Any]:
    items = load_requests()
    result: dict[str, Any] = {"experiment": "E2", "n_requests": len(items), "arms": {}}

    form_rows = [score_request(it, structured_form(form_input(it))) for it in items]
    result["arms"]["structured_form"] = {
        "status": "run",
        "note": "fields entered by hand from the gold labels (non-LLM control)",
        "summary": aggregate(form_rows),
        "rows": form_rows,
    }
    llm_levels: list[dict[str, str]] = []
    if client is None:
        result["arms"]["llm_compiler"] = not_run(llm_reason or "no LLM client configured (run with --llm)")
    else:
        result["meta"] = {"experiment": "E2", **llm_freeze_meta(client)}
        comp = IntentCompiler(client)
        rows, transcripts = [], []
        for it in items:
            res = comp.compile(it["text"])
            rows.append(score_request(it, res))
            transcripts.append({"id": it["id"], "result": res.to_dict(), "transcript": res.transcript})
            if res.request is not None:
                llm_levels.append(res.request["weight_levels"])
        calls = [t["transcript"] for t in transcripts if t["transcript"] is not None]
        failed = sum(1 for t in calls if t.get("error"))
        stats = llm_arm_stats(len(calls), failed, n_proposals=len(calls) - failed,
                              n_valid=sum(1 for r in rows if r["valid"]),
                              account_error=first_account_error(t.get("error") for t in calls))
        arm = {"status": stats["status"], "model": getattr(client, "model", ""), "llm": stats,
               "rows": rows, "transcripts": transcripts}
        if stats["status"] == "run":
            arm["summary"] = aggregate(rows)
        else:
            arm["reason"] = stats.get("reason")
            arm["summary_invalid"] = aggregate(rows)     # kept for audit only; not a result
            llm_levels = []
        result["arms"]["llm_compiler"] = arm
    if direction:
        form_levels = [it["gold"]["weight_levels"] for it in items]
        result["plan_direction"] = {"structured_form": plan_direction(form_levels, scenario, seeds, workers=workers)}
        if llm_levels:
            result["plan_direction"]["llm_compiler"] = plan_direction(llm_levels, scenario, seeds, workers=workers)
    write_json(out, result)
    return result


def llm_client_or_reason(args: argparse.Namespace, cache: bool | None = None) -> tuple[LLMClient | None, str | None]:
    """Preflighted LLM client (DeepSeek by default) only when explicitly requested with --llm.

    Returns ``(client, None)`` or ``(None, reason)``; the reason is written into the not_run arm.
    ``cache`` defaults to ``not --no-cache``.
    """
    if not getattr(args, "llm", False):
        return None, "LLM arm not requested (run with --llm)"
    if cache is None:
        cache = not getattr(args, "no_cache", False)
    return llm_client(cache=cache, provider=getattr(args, "provider", None), model=getattr(args, "model", None))


def llm_client_from_args(args: argparse.Namespace) -> LLMClient | None:
    """Backward-compatible wrapper: the preflighted client or None."""
    return llm_client_or_reason(args)[0]


def add_llm_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--llm", action="store_true", help="run the LLM arm (calls the LLM API; DeepSeek by default)")
    p.add_argument("--provider", default=None, choices=list(PROVIDER_MODELS),
                   help="LLM provider (default: CAMPUSFLEX_PROVIDER or deepseek)")
    p.add_argument("--model", default=None, help="model id (default: provider default, e.g. deepseek-flash)")
    p.add_argument("--no-cache", action="store_true", help="disable the prompt-hash cache")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_llm_args(p)
    p.add_argument("--out", default="results/e2.json")
    p.add_argument("--no-direction", action="store_true")
    p.add_argument("--workers", type=int, default=None)
    a = p.parse_args(argv)
    client, reason = llm_client_or_reason(a)
    if client is None and a.llm:
        print("LLM arm not run:", reason)
    res = run(client, a.out, direction=not a.no_direction, workers=a.workers, llm_reason=reason)
    for arm, v in res["arms"].items():
        print(arm, v["status"], v.get("reason") or json.dumps(v.get("summary", {}).get("success", {})))
    if "plan_direction" in res:
        for arm, v in res["plan_direction"].items():
            print("plan direction", arm, v["intended_share"])


if __name__ == "__main__":
    main()
