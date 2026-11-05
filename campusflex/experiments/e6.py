"""Experiment E6 (RQ6, stretch): does LLM review of development logs (L5) find configuration
changes that improve held-out performance more than random search with the same budget?

Three frozen configurations are compared on the second held-out set, seeds 20–29, over all v2
scenarios: the pre-review configuration used for E1, the L5 result and the random-search
result. If L5 ≈ random search, the gain is credited to extra evaluation budget, not to log
reading (Section 4.7).

Usage:
    python -m campusflex.experiments.e6 review --kind random [--quick]
    python -m campusflex.experiments.e6 review --kind llm [--quick]     # calls the LLM API
    python -m campusflex.experiments.e6 evaluate [--quick]

No LLM result is ever fabricated: without a preflighted client the L5 arm is "not_run", and a
review whose L5 calls mostly failed is "invalid_mostly_fallback" and is not compared. Both
reviews start from, and the pre-review arm is, the frozen configuration used for E1. Arms are
compared in the order of Equation 4: unserved person-minutes first, then the score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config import CHECK_SEEDS, HELDOUT2_SEEDS, TUNING_SEEDS, PlannerConfig
from ..llm.client import PROVIDER_MODELS
from ..roles.l5_reviewer import (
    LLMReviewer,
    RandomSearchProposer,
    ReviewLoop,
    sweep,
    write_report,
)
from ..scenarios import ALL_SCENARIOS
from .common import (
    RESULTS,
    ROOT,
    compare_arms,
    config_hash,
    first_account_error,
    freeze_meta,
    frozen_config,
    llm_arm_stats,
    llm_client,
    llm_freeze_meta,
    write_json,
)

QUICK = {
    "scenarios": ("fault", "notice_surge"),
    "tuning_seeds": (0, 1, 2),
    "check_seeds": (7, 8),
    "heldout_seeds": (20, 21, 22),
    "max_rounds": 2,
}

PATHS = {
    "llm": {"config": RESULTS / "l5_config.json", "history": RESULTS / "l5_history.json",
            "changelog": RESULTS / "l5_config_changelog.json", "report": ROOT / "docs" / "l5_report.md"},
    "random": {"config": RESULTS / "l5_random_config.json", "history": RESULTS / "l5_random_history.json",
               "changelog": RESULTS / "l5_random_changelog.json", "report": RESULTS / "l5_random_report.md"},
}


def review_llm_stats(history: list[dict[str, Any]]) -> dict[str, Any]:
    """L5 health from the proposal history: every proposal is one L5 call."""
    calls = [h for h in history if isinstance(h.get("transcript"), dict)]
    failed = sum(1 for h in calls if h["transcript"].get("error"))
    return llm_arm_stats(len(calls), failed, n_proposals=len(calls) - failed,
                         n_valid=sum(1 for h in calls if h.get("valid")),
                         account_error=first_account_error(h["transcript"].get("error") for h in calls))


def run_review(kind: str = "random", client: Any = None, quick: bool = False, workers: int | None = None,
               seed: int = 0, base_cfg: PlannerConfig | None = None, reason: str | None = None) -> dict[str, Any]:
    """Run one review loop (L5 or the random-search control) and freeze its configuration."""
    fcfg, fh, ffrozen = frozen_config()
    base_cfg = base_cfg or fcfg
    bh = config_hash(base_cfg)
    if kind == "llm":
        if client is None:
            return {"status": "not_run", "reason": reason or "no LLM client configured"}
        proposer = LLMReviewer(client)
    elif kind == "random":
        proposer = RandomSearchProposer(seed)
    else:
        raise ValueError(kind)
    p = PATHS[kind]
    for f in ("changelog",):
        if p[f].exists():
            p[f].unlink()          # a fresh review starts a fresh changelog
    loop = ReviewLoop(
        proposer,
        base_cfg,
        scenarios=QUICK["scenarios"] if quick else ALL_SCENARIOS,
        tuning_seeds=QUICK["tuning_seeds"] if quick else TUNING_SEEDS,
        check_seeds=QUICK["check_seeds"] if quick else CHECK_SEEDS,
        max_rounds=QUICK["max_rounds"] if quick else 6,
        # L5 stops early after two empty rounds (Section 3.4); the random-search control spends the
        # full budget of 18 proposals ("the same budget"), so it never has less budget than L5.
        stop_after_empty_rounds=2 if kind == "llm" else None,
        workers=workers,
        out_config=p["config"],
        changelog=p["changelog"],
        label="L5" if kind == "llm" else "random",
    )
    outcome = loop.run()
    outcome.meta["protocol"] = "reduced" if quick else "full"
    outcome.meta.update({"base_config_hash": bh, "base_frozen": ffrozen and bh == fh})
    if kind == "llm":
        outcome.meta.update(llm_freeze_meta(client))
    hist = outcome.to_dict()
    if kind == "llm":
        stats = review_llm_stats(hist.get("history", []))
        hist["llm_stats"] = stats
        hist["status"] = stats["status"]
        if stats["status"] != "run":
            hist["reason"] = stats.get("reason")
    else:
        hist["status"] = "run"
    write_json(p["history"], hist)
    rep = write_report(outcome, p["report"])
    return {"status": hist["status"], "protocol": outcome.meta["protocol"], "accepted": len(outcome.accepted),
            "proposals": outcome.proposals, "sweeps": outcome.sweeps, "config": str(p["config"]),
            "report": rep, **({"llm": hist["llm_stats"]} if "llm_stats" in hist else {})}


def _load_cfg(path: Path) -> PlannerConfig | None:
    return PlannerConfig.load(path) if path.exists() else None


def _history(kind: str) -> dict[str, Any] | None:
    h = PATHS[kind]["history"]
    if not h.exists():
        return None
    try:
        d = json.loads(h.read_text())
    except (ValueError, OSError):
        return None
    return d if isinstance(d, dict) else None


def _history_protocol(kind: str) -> str | None:
    d = _history(kind)
    return (d or {}).get("meta", {}).get("protocol")


def _review_arm(kind: str, pre_hash: str) -> dict[str, Any]:
    """Status of a reviewed configuration: it needs its config file and a healthy review history
    that started from the pre-review (frozen) configuration."""
    c = _load_cfg(PATHS[kind]["config"])
    if c is None:
        return {"status": "not_run", "reason": f"{PATHS[kind]['config'].name} missing; run `review --kind {kind}`"}
    h = _history(kind)
    if h is None:
        return {"status": "not_run", "reason": f"{PATHS[kind]['history'].name} missing: review provenance unknown"}
    status = h.get("status", "run")
    if status != "run":
        return {"status": status, "reason": h.get("reason") or status, "llm": h.get("llm_stats")}
    base = (h.get("meta") or {}).get("base_config_hash")
    if base and base != pre_hash:
        return {"status": "invalid_base_config_mismatch",
                "reason": f"review started from config {base}, pre-review arm is {pre_hash}"}
    hm = h.get("meta") or {}
    return {"status": "run", "config": c.to_dict(), "config_hash": config_hash(c),
            "source": str(PATHS[kind]["config"]), "review_protocol": hm.get("protocol"),
            **({"llm": h["llm_stats"]} if "llm_stats" in h else {}),
            **{k: hm[k] for k in ("prompts_frozen", "prompts_note", "prompt_hashes", "model") if k in hm}}


def evaluate(quick: bool = False, workers: int | None = None, policy: str = "arm_B") -> dict[str, Any]:
    scenarios = QUICK["scenarios"] if quick else ALL_SCENARIOS
    seeds = QUICK["heldout_seeds"] if quick else HELDOUT2_SEEDS
    pre_cfg, pre_h, frozen = frozen_config()
    arms: dict[str, dict[str, Any]] = {
        "pre_review": {"status": "run", "config": pre_cfg.to_dict(), "config_hash": pre_h,
                       "source": "results/frozen_config.json" if frozen else "PlannerConfig() defaults (not frozen)"},
        "L5": _review_arm("llm", pre_h),
        "random": _review_arm("random", pre_h),
    }
    rows: list[dict[str, Any]] = []
    for name, a in arms.items():
        if a["status"] != "run":
            continue
        cfg = PlannerConfig.from_dict(a["config"])
        for r in sweep(cfg, seeds, scenarios, workers=workers, policy=policy):
            r["arm"] = name
            r["config_hash"] = a["config_hash"]
            r.setdefault("mismatch", False)
            rows.append(r)

    def paired(x: str, y: str) -> dict[str, Any] | None:
        if arms[x]["status"] != "run" or arms[y]["status"] != "run":
            return None
        c = compare_arms(rows, "arm", x, y)
        return {k: v for k, v in c.items() if k != "keys"} if c else None

    comps = {
        "L5_minus_pre_review": paired("L5", "pre_review"),
        "random_minus_pre_review": paired("random", "pre_review"),
        "L5_minus_random": paired("L5", "random"),
    }
    verdict: dict[str, Any] = {"success_criterion": "L5 beats pre-review with interval excluding zero and is not worse "
                                                    "than random search (Eq. 4 order: unserved first, then score)"}
    c1, c2 = comps["L5_minus_pre_review"], comps["L5_minus_random"]
    if c1 is None or c2 is None:
        verdict["result"] = "not evaluable (L5 or random-search arm not run or invalid)"
    else:
        beats = c1["a_better"]
        not_worse = not c2["a_worse"]
        verdict.update({"L5_beats_pre_review": beats, "L5_not_worse_than_random": not_worse,
                        "decided_by": {"L5_vs_pre_review": c1["decided_by"], "L5_vs_random": c2["decided_by"]},
                        "success": bool(beats and not_worse)})
        if not (c2["a_better"] or c2["a_worse"]):
            verdict["falsification"] = ("L5 ≈ random search: the gain comes from spending more evaluation budget "
                                        "on development seeds, not from reading logs.")
    reviews = {}
    for kind in ("llm", "random"):
        d = _history(kind)
        if d is not None:
            reviews[kind] = {"status": d.get("status", "run"), "review_acceptance_rate": d.get("review_acceptance_rate"),
                             "proposals": d.get("proposals"), "accepted": len(d.get("accepted", [])),
                             "sweeps": d.get("sweeps"), "protocol": d.get("meta", {}).get("protocol"),
                             **({"llm": d["llm_stats"]} if "llm_stats" in d else {})}
    per_arm = {}
    for name in arms:
        rs = [r for r in rows if r["arm"] == name]
        if rs:
            per_arm[name] = {m: float(np.mean([r[m] for r in rs])) for m in
                             ("unserved_person_min", "score", "kwh", "dev_c", "moves", "switches")}
    l5 = arms["L5"]
    l5_prompt = ({"prompts_frozen": False, "prompts_note": f"L5 review: {l5.get('prompts_note')}"}
                 if l5.get("status") == "run" and l5.get("prompts_frozen") is False else {})
    # A reduced review (``review --quick``) makes the whole E6 result reduced, not only ``evaluate --quick``.
    reduced_reviews = [k for k, r in reviews.items() if r.get("protocol") == "reduced"]
    budget = {k: {"proposals": r.get("proposals"), "sweeps": r.get("sweeps")} for k, r in reviews.items()}
    if {"llm", "random"} <= set(budget) and (budget["random"]["proposals"] or 0) < (budget["llm"]["proposals"] or 0):
        verdict["budget_note"] = (f"random search used fewer proposals ({budget['random']['proposals']}) than L5 "
                                  f"({budget['llm']['proposals']}): the control had less evaluation budget")
    out = {
        "meta": {"experiment": "E6", "protocol": "reduced" if quick or reduced_reviews else "full",
                 **({"reduced_reviews": reduced_reviews} if reduced_reviews else {}), "budget": budget,
                 "policy": policy, **l5_prompt,
                 "scenarios": list(scenarios), "seeds": list(seeds), **freeze_meta(pre_h, frozen),
                 "order": "Eq. 4: unserved person-minutes first, then score",
                 "note": "evaluation arm is the non-LLM pre-review arm (masks + rule-staged); L2 trigger "
                         "parameters therefore have no effect in this evaluation"},
        "arms": arms,
        "per_arm_means": per_arm,
        "comparisons": comps,
        "verdict": verdict,
        "reviews": reviews,
        "rows": rows,
    }
    write_json(RESULTS / "e6.json", out)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("review")
    r.add_argument("--kind", choices=["random", "llm"], default="random")
    r.add_argument("--quick", action="store_true", help="reduced protocol (labelled as such)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--workers", type=int, default=None)
    r.add_argument("--model", default=None, help="LLM model id (kind=llm)")
    r.add_argument("--provider", default=None, choices=list(PROVIDER_MODELS),
                   help="LLM provider (kind=llm; default: CAMPUSFLEX_PROVIDER or deepseek)")
    e = sub.add_parser("evaluate")
    e.add_argument("--quick", action="store_true", help="reduced protocol (labelled as such)")
    e.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "review":
        client, reason = None, None
        if a.kind == "llm":
            client, reason = llm_client(cache=True, provider=a.provider, model=a.model)
            if client is None:
                print("L5 review not run:", reason)
        res = run_review(a.kind, client=client, quick=a.quick, workers=a.workers, seed=a.seed, reason=reason)
        print(json.dumps(res, indent=1, default=str))
    else:
        out = evaluate(quick=a.quick, workers=a.workers)
        print(json.dumps({k: out[k] for k in ("meta", "per_arm_means", "comparisons", "verdict")}, indent=1, default=str))


if __name__ == "__main__":
    main()
