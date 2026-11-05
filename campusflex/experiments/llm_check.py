"""LLM readiness check: ``python -m campusflex llm-check`` (run once after setting DEEPSEEK_API_KEY).

Before hours of live runs, this sends each role's real prompt a few times, uncached, through the
same budgeted ``call`` path the experiments use, on development-seed inputs only (the E5 fixture
state: fault scenario, seed 0, 20:00; no held-out seed is touched). Per role it reports success,
error kinds, latency against the 20 s step budget, tokens (with DeepSeek prefix-cache hits) and
whether the output is usable (L2: plans that pass the validator; L1: a compiled request; L5: a
proposal that passes the Table 9 check). It then projects the E1 C/D workload: L2 calls, wall
time at the chosen thread count, and tokens (cost only if prices are given).

    python -m campusflex llm-check [--repeats 3] [--threads 8] [--price-in USD --price-out USD]

Output: ``results/llm_check.json``. This is a readiness check, not evidence for any ledger item.
"""
from __future__ import annotations

import argparse
import statistics
from collections import Counter
from typing import Any, Callable

from ..config import PlannerConfig, parse_clock
from ..llm.client import PROVIDER_MODELS, CallBudget, ScriptedClient, call
from ..scenarios import Notice, make_scenario
from .common import llm_freeze_meta, write_json

ROLES = ("L1", "L2", "L3", "L4_explain", "L5")
STEP_ROLES = ("L2", "L3")                   # called inside a decision step: 20 s budget applies
NEAR_BUDGET = 0.75                          # warn when the median latency exceeds this share of the budget
DEMO_REQUEST = ("Provide space for 120 students this evening, maintain comfortable conditions, use as little "
                "electricity as practical, and avoid moving students who are already seated.")


def _fixture(cfg: PlannerConfig):
    from .e5 import build_fixture_env

    env = build_fixture_env(cfg)
    return env.get_state()


def _l2_planner(client: Any, cfg: PlannerConfig):
    from ..policies import make_policy
    from ..roles.l2_proposer import LLMStagedProposer

    pol = make_policy("arm_C", cfg, l2=LLMStagedProposer(client, cfg))
    pol.reset()
    return pol


def probe_l2(client: Any, cfg: PlannerConfig, snap) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    pol = _l2_planner(client, cfg)
    dec = pol.decide(snap)
    trs = [t for t in dec.llm if t.get("role") == "L2" and "fallback" not in t]
    st = pol.stats
    return (trs[-1] if trs else None), {"plans": st["l2_plans"], "valid_plans": st["l2_valid"],
                                        "usable": st["l2_valid"] > 0, "_ranking": dec.ranking}


def probe_l1(client: Any, cfg: PlannerConfig, snap) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    from ..roles.l1_intent import IntentCompiler

    res = IntentCompiler(client).compile(DEMO_REQUEST)
    return res.transcript, {"usable": res.request is not None, "questions": len(res.questions),
                            "errors": res.errors[:3]}


def probe_l3(client: Any, cfg: PlannerConfig, snap) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    from ..roles.l3_events import LLMInterpreter
    from .e5_fixtures import N1_TEXT

    interp = LLMInterpreter(client, cfg, CallBudget(cfg.llm_calls_per_step, cfg.llm_latency_s))
    adjs = interp.interpret([Notice("N1", parse_clock("20:00"), N1_TEXT)], snap)
    tr = interp.transcripts[-1] if interp.transcripts else None
    return tr, {"usable": any(a.get("kind") not in (None, "none") for a in adjs), "adjustments": len(adjs)}


def probe_l4_explain(client: Any, cfg: PlannerConfig, snap, ranking: list[dict[str, Any]] | None = None
                     ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    from .e4 import EXPLAIN_SCHEMA, EXPLAIN_SYSTEM, compact_ranking, explain_user_prompt

    if ranking is None:
        from ..policies import make_policy

        pol = make_policy("arm_A", cfg)
        pol.reset()
        ranking = pol.decide(snap).ranking
    logged = {"ranking": compact_ranking(ranking), "horizon_min": cfg.rollout_horizon_min}
    r = call(client, CallBudget(max_calls=10**9, latency_s=float("inf")), role="L4", system=EXPLAIN_SYSTEM,
             user=explain_user_prompt(logged), schema=EXPLAIN_SCHEMA)
    return r.transcript(), {"usable": r.ok and isinstance(r.data, dict) and isinstance(r.data.get("explanation"), str)}


_DIGEST: dict[str, Any] = {}


def probe_l5(client: Any, cfg: PlannerConfig, snap) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    from ..roles.l5_reviewer import LLMReviewer, build_digest, check_proposal, sweep

    if "d" not in _DIGEST:                   # one small development sweep (non-LLM), reused across repeats
        _DIGEST["d"] = build_digest(sweep(cfg, (0,), ("fault",), keep_steps=True, workers=1), cfg, [])
    digest = _DIGEST["d"]
    res = LLMReviewer(client).propose(digest, cfg, [], "R1")
    issues = check_proposal(res.proposal, digest, cfg, []) if res.proposal is not None else []
    return res.transcript, {"usable": res.proposal is not None and not issues,
                            "issues": [i.to_dict() for i in issues][:3]}


PROBES: dict[str, Callable[..., tuple[dict[str, Any] | None, dict[str, Any]]]] = {
    "L1": probe_l1, "L2": probe_l2, "L3": probe_l3, "L4_explain": probe_l4_explain, "L5": probe_l5,
}


def _mean(xs: list[float]) -> float | None:
    return round(float(statistics.fmean(xs)), 1) if xs else None


def summarize_role(role: str, runs: list[tuple[dict[str, Any] | None, dict[str, Any]]],
                   budget_s: float) -> dict[str, Any]:
    trs = [t for t, _ in runs]
    made = [t for t in trs if t]
    # token means over calls that reported usage: a timed-out call is billed but reports none, and
    # every projected E1 C/D call is billed
    billed = [t for t in made if (t.get("input_tokens") or 0) > 0]
    errors = Counter(t.get("error") or "no_call" for t in trs if not t or t.get("error"))
    lat = [float(t.get("latency_s") or 0.0) for t in made]
    out: dict[str, Any] = {
        "n": len(runs),
        "ok_calls": sum(1 for t in made if not t.get("error")),
        "usable_outputs": sum(1 for _, x in runs if x.get("usable")),
        "errors": dict(errors),
        "latency_s": {"median": round(statistics.median(lat), 2) if lat else None,
                      "max": round(max(lat), 2) if lat else None},
        "input_tokens_mean": _mean([t.get("input_tokens") or 0 for t in billed]),
        "output_tokens_mean": _mean([t.get("output_tokens") or 0 for t in billed]),
        "cache_hit_tokens_mean": _mean([t.get("cache_hit_tokens") or 0 for t in billed]),
        "stop_reasons": dict(Counter(str(t.get("stop_reason")) for t in made)),
        "models": sorted({t.get("model") for t in made if t.get("model")}),
        "system_fingerprints": sorted({t.get("system_fingerprint") for t in made if t.get("system_fingerprint")}),
    }
    if role in STEP_ROLES:
        out["step_budget_s"] = budget_s
        out["within_step_budget"] = sum(1 for t in made if not t.get("error") and float(t.get("latency_s") or 0) <= budget_s)
    if role == "L2":
        out["valid_plans_mean"] = _mean([x.get("valid_plans", 0) for _, x in runs])
        out["plans_mean"] = _mean([x.get("plans", 0) for _, x in runs])
    extras = [{k: v for k, v in x.items() if k not in ("usable", "_ranking")} for _, x in runs]
    out["details"] = extras[:3]
    return out


def count_e1_cd_l2_calls(cfg: PlannerConfig, seed: int = 0) -> dict[str, Any]:
    """L2 trigger count per E1 C/D episode, measured with an always-empty LLM on development seed
    ``seed`` across the E1 grid (scenarios × mismatch × arms). Real replies change the trajectory,
    so the projection is approximate."""
    from ..episode import run_episode
    from ..policies import make_policy
    from ..roles.l2_proposer import LLMStagedProposer
    from .e1 import LLM_ARMS, _grid

    scen, mms, repeats = _grid(False)
    per: list[int] = []
    for arm in LLM_ARMS:
        for s in scen:
            for mm in mms:
                pol = make_policy(arm, cfg, l2=LLMStagedProposer(ScriptedClient(lambda *_: {"plans": []}), cfg))
                r = run_episode(make_scenario(s, seed, mismatch=mm), pol, cfg)
                per.append(int(r.policy_stats.get("l2_calls", 0)))
    from ..config import HELDOUT_SEEDS

    n_eps = len(per) * len(HELDOUT_SEEDS) * repeats
    mean = sum(per) / len(per)
    return {"l2_calls_per_episode_mean": round(mean, 2), "episodes": n_eps, "l2_calls_est": int(round(mean * n_eps)),
            "measured_on": f"development seed {seed}, scripted empty replies"}


def project(roles: dict[str, Any], counts: dict[str, Any], threads: int, prices: dict[str, float | None]
            ) -> dict[str, Any]:
    l2 = roles.get("L2") or {}
    n = counts["l2_calls_est"]
    lat = (l2.get("latency_s") or {}).get("median")
    tin, tout, thit = (l2.get(k) or 0.0 for k in ("input_tokens_mean", "output_tokens_mean", "cache_hit_tokens_mean"))
    out: dict[str, Any] = {**counts, "threads": threads,
                           "wall_hours_est": round(n * lat / max(1, threads) / 3600, 2) if lat else None,
                           "input_tokens_est": int(n * tin), "output_tokens_est": int(n * tout),
                           "cache_hit_tokens_est": int(n * thit)}
    if prices.get("in") is not None and prices.get("out") is not None:
        hit_price = prices["hit"] if prices.get("hit") is not None else prices["in"]
        out["cost_usd_est"] = round((n * (tin - thit) * prices["in"] + n * thit * hit_price
                                     + n * tout * prices["out"]) / 1e6, 2)
        out["prices_usd_per_mtok"] = {"input": prices["in"], "input_cache_hit": hit_price, "output": prices["out"]}
    return out


def verdict(roles: dict[str, Any], budget_s: float, freeze: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(warnings, notes). A role fails the check when at least half of its calls fail or are unusable
    (the E1 arms are invalid above 50 % fallback) or on any account error; an occasional bad reply
    (live DeepSeek: ~3 % invalid JSON) is only a note – the planner falls back for that step."""
    from ..llm.client import is_account_error

    warn, notes = [], []
    for role, s in roles.items():
        n, bad, unusable = s["n"], s["n"] - s["ok_calls"], s["n"] - s["usable_outputs"]
        if any(is_account_error(e) for e in s["errors"]):
            warn.append(f"{role}: account error {s['errors']} – fix the API key / balance before any run")
        elif bad:
            (warn if 2 * bad >= n else notes).append(f"{role}: {bad}/{n} calls failed ({s['errors']})")
        elif unusable:
            (warn if 2 * unusable >= n else notes).append(f"{role}: {unusable}/{n} replies were not usable "
                                                          f"(see details)")
        med = (s.get("latency_s") or {}).get("median")
        if role in STEP_ROLES and med is not None:
            if med > budget_s:
                warn.append(f"{role}: median latency {med:.1f} s exceeds the {budget_s:.0f} s step budget; most calls "
                            "would fall back and the LLM arms would be invalid_mostly_fallback")
            elif med > NEAR_BUDGET * budget_s:
                warn.append(f"{role}: median latency {med:.1f} s is close to the {budget_s:.0f} s step budget; "
                            "expect fallbacks under load")
    if freeze.get("prompts_frozen") is False:
        warn.append(f"freeze gate: {freeze.get('prompts_note')}")
    return warn, notes


def run(client: Any, cfg: PlannerConfig | None = None, repeats: int = 3, roles: tuple[str, ...] = ROLES,
        threads: int = 8, prices: dict[str, float | None] | None = None, projection: bool = True,
        out: str | None = "results/llm_check.json") -> dict[str, Any]:
    from .common import frozen_config

    cfg = cfg or frozen_config()[0]
    snap = _fixture(cfg)
    budget_s = float(cfg.llm_latency_s)
    per_role: dict[str, Any] = {}
    ranking = None
    for role in roles:
        runs = []
        for _ in range(max(1, repeats)):
            if role == "L4_explain":
                tr, x = probe_l4_explain(client, cfg, snap, ranking)
            else:
                tr, x = PROBES[role](client, cfg, snap)
            if role == "L2" and ranking is None:
                ranking = x.get("_ranking")
            runs.append((tr, x))
        per_role[role] = summarize_role(role, runs, budget_s)
    freeze = llm_freeze_meta(client)
    res: dict[str, Any] = {
        "meta": {"experiment": "LLM readiness check (not ledger evidence)", "model": getattr(client, "model", None),
                 "repeats": repeats, "inputs": "E5 fixture state (fault, development seed 0, 20:00); no held-out seed",
                 "caching": "disabled", "step_budget_s": budget_s, "calls_per_step": cfg.llm_calls_per_step,
                 **{k: freeze[k] for k in ("prompts_frozen", "prompts_note") if k in freeze}},
        "roles": per_role,
    }
    if projection and "L2" in per_role:
        res["projection_e1_cd"] = project(per_role, count_e1_cd_l2_calls(cfg), threads, prices or {})
    res["warnings"], res["notes"] = verdict(per_role, budget_s, freeze)
    res["status"] = "ready" if not res["warnings"] else "check_warnings"
    if out:
        write_json(out, res)
    return res


def _print(res: dict[str, Any]) -> None:
    print(f"model {res['meta'].get('model')}  status: {res['status']}")
    print(f"{'role':<11}{'ok':>6}{'usable':>8}{'median s':>10}{'max s':>8}{'in tok':>8}{'out tok':>9}{'cache hit':>10}")
    for role, s in res["roles"].items():
        lat = s["latency_s"]
        print(f"{role:<11}{s['ok_calls']:>3}/{s['n']:<2}{s['usable_outputs']:>6}/{s['n']:<2}"
              f"{lat['median'] if lat['median'] is not None else '-':>9}{lat['max'] if lat['max'] is not None else '-':>8}"
              f"{s['input_tokens_mean'] or '-':>8}{s['output_tokens_mean'] or '-':>9}{s['cache_hit_tokens_mean'] or '-':>10}")
    p = res.get("projection_e1_cd")
    if p:
        print(f"E1 C/D projection: ~{p['l2_calls_est']} L2 calls over {p['episodes']} episodes"
              + (f", ~{p['wall_hours_est']} h at {p['threads']} threads" if p.get("wall_hours_est") is not None else "")
              + f", ~{p['input_tokens_est'] / 1e6:.1f}M input / {p['output_tokens_est'] / 1e6:.1f}M output tokens"
              + (f", ~${p['cost_usd_est']}" if "cost_usd_est" in p else ""))
    for w in res["warnings"]:
        print("WARNING:", w)
    for w in res.get("notes") or []:
        print("NOTE:", w)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default=None, choices=list(PROVIDER_MODELS))
    ap.add_argument("--model", default=None)
    ap.add_argument("--repeats", type=int, default=3, help="calls per role (default 3)")
    ap.add_argument("--roles", default=",".join(ROLES), help=f"comma-separated subset of {','.join(ROLES)}")
    ap.add_argument("--threads", type=int, default=8, help="E1 C/D thread count for the wall-time projection")
    ap.add_argument("--price-in", type=float, default=None, help="USD per 1M input tokens (cache miss)")
    ap.add_argument("--price-cache-hit", type=float, default=None, help="USD per 1M input tokens served from cache")
    ap.add_argument("--price-out", type=float, default=None, help="USD per 1M output tokens")
    ap.add_argument("--no-projection", action="store_true")
    ap.add_argument("--out", default="results/llm_check.json")
    a = ap.parse_args(argv)
    roles = tuple(r.strip() for r in a.roles.split(",") if r.strip())
    unknown = [r for r in roles if r not in ROLES]
    if unknown:
        raise SystemExit(f"unknown role(s) {unknown}; choose from {ROLES}")
    from .common import llm_client

    client, reason = llm_client(cache=False, provider=a.provider, model=a.model)
    if client is None:
        write_json(a.out, {"meta": {"experiment": "LLM readiness check"}, "status": "not_run", "reason": reason})
        raise SystemExit(f"LLM not reachable: {reason}")
    res = run(client, repeats=a.repeats, roles=roles, threads=a.threads, projection=not a.no_projection,
              prices={"in": a.price_in, "out": a.price_out, "hit": a.price_cache_hit}, out=a.out)
    _print(res)
    if res["status"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
