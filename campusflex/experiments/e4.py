"""E4 (RQ5): faithfulness of explanations – LLM explanations vs template text.

For sampled decision steps, an explanation of the committed choice is produced from the
planner's logged ranking (tool values). The faithfulness checker extracts every number and
matches it against the logged values; unmatched numbers are removed and flagged before
display. Metric: share of numbers matching a logged tool value. Success criterion: 100 %
matched or flagged before display.

Decisions come from saved episode files (``--episodes``: JSON with a ``decisions`` list, as
written by ``EpisodeResult.save``) or are generated fresh with the masks-only planner on the
E1 and E3 scenarios under the frozen configuration. The LLM arm runs only with ``--llm`` and
after a real preflight call; an arm whose calls mostly failed is "invalid_mostly_fallback". No
success is reported for an arm with zero evaluated texts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config import HELDOUT_SEEDS, PlannerConfig
from ..llm.client import CallBudget, LLMClient, call
from ..roles.faithfulness import check
from ..scenarios import E1_SCENARIOS, NOTICE_SCENARIOS
from .common import (
    config_hash,
    episode_job,
    first_account_error,
    freeze_meta,
    frozen_config,
    llm_arm_stats,
    llm_freeze_meta,
    not_run,
    run_parallel,
    write_json,
)
from .e2 import add_llm_args, llm_client_or_reason

EXPLAIN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"explanation": {"type": "string"}},
    "required": ["explanation"],
    "additionalProperties": False,
}

EXPLAIN_SYSTEM = """You explain the decisions of CampusFlex, a controller for six study rooms, to the \
building administrator. You receive the planner's ranking of candidate plans; every number in it was \
computed by the planner's simulator (E = kWh over the rollout, dev_mean = mean occupied deviation from \
the comfort band in °C, M = student moves, K = room switches, U = unserved person-minutes, \
score_J = weighted score, lower is better).
Write 2-4 plain sentences explaining why the top-ranked plan was chosen over the runner-up. Use only \
numbers that appear in the ranking JSON, written exactly as they appear; do not compute differences, \
sums or percentages. Every number is checked automatically and unmatched numbers are removed."""

KEEP = ("rank", "plan_id", "source", "E", "V", "M", "K", "U", "Omega", "dev_mean", "score_J")


def compact_ranking(ranking: list[dict[str, Any]], top: int = 3) -> list[dict[str, Any]]:
    return [{k: r[k] for k in KEEP if k in r} for r in ranking[:top]]


def template_explanation(ranking: list[dict[str, Any]], horizon_min: int) -> str:
    a = ranking[0]
    txt = (f"The planner chose {a['plan_id']} ({a['source']}): predicted {a['E']} kWh over the next "
           f"{horizon_min} minutes, mean occupied deviation {a['dev_mean']} °C, {a['M']} moves and "
           f"{a['K']} room switches.")
    if len(ranking) > 1:
        b = ranking[1]
        txt += (f" The runner-up {b['plan_id']} would use {b['E']} kWh with deviation {b['dev_mean']} °C and "
                f"{b['M']} moves; its score is {b['score_J']} against {a['score_J']}.")
    return txt


def sample_decisions(decisions: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    cands = [d for d in decisions if len(d.get("ranking") or []) >= 2]
    if len(cands) <= k:
        return cands
    idx = np.linspace(0, len(cands) - 1, k).round().astype(int)
    return [cands[i] for i in sorted(set(idx.tolist()))]


def load_decisions(paths: list[str]) -> list[dict[str, Any]]:
    out = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        episodes = d if isinstance(d, list) else [d]
        for ep in episodes:
            decs = ep.get("decisions") or ep.get("_decisions") or []
            tag = ep.get("summary", {}) if isinstance(ep.get("summary"), dict) else ep
            out.append({"episode": f"{tag.get('scenario')}/s{tag.get('seed')}/{tag.get('policy')}", "decisions": decs})
    return out


def fresh_decisions(scenarios: tuple[str, ...], seeds: tuple[int, ...], cfg: PlannerConfig,
                    workers: int | None = None) -> list[dict[str, Any]]:
    specs = [{"scenario": s, "seed": sd, "policy": "arm_A", "cfg": cfg.to_dict(), "keep_decisions": True}
             for s in scenarios for sd in seeds]
    rows = run_parallel(episode_job, specs, workers)
    return [{"episode": f"{r['scenario']}/s{r['seed']}/A", "decisions": r.get("_decisions", [])} for r in rows]


def evaluate(texts: list[dict[str, Any]]) -> dict[str, Any]:
    n_num = sum(len(t["faithfulness"]["numbers"]) for t in texts)
    n_match = sum(len(t["faithfulness"]["matched"]) for t in texts)
    n_flag = sum(len(t["faithfulness"]["flagged"]) for t in texts)
    shown_unverified = sum(t["displayed_unverified"] for t in texts)
    return {
        "n_texts": len(texts),
        "n_numbers": n_num,
        "n_matched": n_match,
        "n_flagged": n_flag,
        "share_matched": (n_match / n_num) if n_num else None,
        "texts_fully_matched_share": (sum(1 for t in texts if not t["faithfulness"]["flagged"]) / len(texts)) if texts else None,
        "displayed_unverified_numbers": shown_unverified,
        # no texts -> nothing was shown, which is not evidence of faithfulness
        "success_100pct_matched_or_flagged": (shown_unverified == 0) if texts else None,
    }


def explain_user_prompt(logged: dict[str, Any]) -> str:
    """User prompt of the E4 explanation call (fingerprinted at the freeze gate)."""
    return "Planner ranking (tool output):\n" + json.dumps(logged, indent=1)


def _record(ep: str, d: dict[str, Any], text: str, logged: Any, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    fr = check(text, logged)
    recheck = check(fr.cleaned_text, logged)           # what the administrator actually sees
    return {"episode": ep, "step": d.get("step"), "text": text, "displayed": fr.cleaned_text,
            "faithfulness": fr.to_dict(), "displayed_unverified": len(recheck.flagged), **(extra or {})}


def run(client: LLMClient | None = None, out: str | Path = "results/e4.json", episodes: list[str] | None = None,
        scenarios: tuple[str, ...] = E1_SCENARIOS + NOTICE_SCENARIOS, seeds: tuple[int, ...] = HELDOUT_SEEDS[:2],
        per_episode: int = 3, workers: int | None = None, cfg: PlannerConfig | None = None,
        llm_reason: str | None = None) -> dict[str, Any]:
    fcfg, fh, ffrozen = frozen_config()
    cfg = cfg or fcfg
    h = config_hash(cfg)
    src = load_decisions(episodes) if episodes else fresh_decisions(scenarios, seeds, cfg, workers)
    tmpl, llm = [], []
    budget = CallBudget(max_calls=10**9, latency_s=float("inf"))
    for ep in src:
        for d in sample_decisions(ep["decisions"], per_episode):
            rk = compact_ranking(d["ranking"])
            logged = {"ranking": rk, "horizon_min": cfg.rollout_horizon_min}
            tmpl.append(_record(ep["episode"], d, template_explanation(rk, cfg.rollout_horizon_min), logged))
            if client is not None:
                r = call(client, budget, role="L4", system=EXPLAIN_SYSTEM, user=explain_user_prompt(logged),
                         schema=EXPLAIN_SCHEMA)
                if r.ok and isinstance(r.data, dict) and isinstance(r.data.get("explanation"), str):
                    llm.append(_record(ep["episode"], d, r.data["explanation"], logged,
                                       {"model": r.model, "latency_s": r.latency_s,
                                        "input_tokens": r.input_tokens, "output_tokens": r.output_tokens}))
                else:
                    llm.append({"episode": ep["episode"], "step": d.get("step"), "error": r.error})
    result: dict[str, Any] = {
        "experiment": "E4",
        "meta": {"experiment": "E4", **freeze_meta(h, ffrozen and h == fh), "seeds": None if episodes else list(seeds),
                 "scenarios": None if episodes else list(scenarios), "per_episode": per_episode,
                 **(llm_freeze_meta(client) if client is not None else {})},
        "source": "episode files" if episodes else "fresh masks-only planner decisions",
        "n_episodes": len(src),
        "arms": {"template": {"status": "run", "summary": evaluate(tmpl), "examples": tmpl[:5]}},
    }
    if client is None:
        result["arms"]["llm"] = not_run(llm_reason or "no LLM client configured (run with --llm)")
    else:
        ok = [t for t in llm if "faithfulness" in t]
        stats = llm_arm_stats(len(llm), len(llm) - len(ok), n_proposals=len(ok),
                              n_valid=sum(1 for t in ok if not t["faithfulness"]["flagged"]),
                              account_error=first_account_error(t.get("error") for t in llm))
        arm = {"status": stats["status"], "model": getattr(client, "model", ""), "llm": stats,
               "n_failed_calls": len(llm) - len(ok), "examples": ok[:5], "texts": llm}
        if stats["status"] == "run":
            arm["summary"] = evaluate(ok)
        else:
            arm["reason"] = stats.get("reason")
            arm["summary_invalid"] = evaluate(ok)        # audit only; not a result
        result["arms"]["llm"] = arm
    result["arms"]["template"]["texts"] = tmpl
    write_json(out, result)
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_llm_args(p)
    p.add_argument("--out", default="results/e4.json")
    p.add_argument("--episodes", nargs="*", default=None, help="episode JSON files with decisions")
    p.add_argument("--per-episode", type=int, default=3)
    p.add_argument("--workers", type=int, default=None)
    a = p.parse_args(argv)
    client, reason = llm_client_or_reason(a)
    if client is None and a.llm:
        print("LLM arm not run:", reason)
    res = run(client, a.out, a.episodes, per_episode=a.per_episode, workers=a.workers, llm_reason=reason)
    print(json.dumps({k: {"status": v["status"], **({"summary": v["summary"]} if "summary" in v else {}),
                          **({"reason": v["reason"]} if v.get("reason") else {})}
                      for k, v in res["arms"].items()}, indent=1))


if __name__ == "__main__":
    main()
