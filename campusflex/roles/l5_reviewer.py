"""L5 episode reviewer (stretch; Section 3.4, Table 9, experiment E6).

L5 turns patterns in development logs into bounded configuration changes. It acts between
runs, never inside one, and its failure mode is overfitting, so the design:

* restricts it to single-parameter, one-step diffs inside the declared Table 9 space;
* gives it a deterministic *digest* of a development sweep (never raw logs), in which every
  interval carries an identifier such as ``fault/s4/20:00-20:40``;
* requires evidence that can be checked: cited intervals must exist in the digest, the
  ``from`` value must equal the current configuration, observations may not contain numbers,
  and a diff that was already rejected may not be proposed again;
* accepts a diff only if, on tuning seeds 0–6, the mean full-evening score improves with a
  paired bootstrap 95% interval excluding zero, unserved person-minutes and hard-constraint
  violations do not increase in any scenario, and a confirmation sweep on check seeds 7–9 does
  not worsen the mean score;
* compares it with a random-search control that draws one-step diffs from the same space and
  applies the same acceptance rule with the same budget (6 rounds × 3 proposals = 18).

Interval construction (documented choice): each episode's 24 committed steps are cut into
fixed 40-minute windows (4 steps: 18:00-18:40, …, 21:20-22:00). A window's contribution is
Equation 4 without Ω under the reference weights, Σ (E + 8 V + 0.06 M + 0.08 K), so the
digest ranks the same quantity the acceptance rule measures. The full-evening score always
uses the reference weights (``metrics.REFERENCE_WEIGHTS``), so lowering λT cannot game it.
"""
from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

import numpy as np

from ..config import (
    CHECK_SEEDS,
    L5_PARAM_SPACE,
    STEP_MIN,
    TUNING_SEEDS,
    PlannerConfig,
    clock_str,
    parse_clock,
)
from ..metrics import REFERENCE_WEIGHTS
from ..scenarios import ALL_SCENARIOS
from ..stats import paired_bootstrap
from .faithfulness import check as faithfulness_check

ROOT = Path(__file__).resolve().parents[2]
WINDOW_STEPS = 4                     # 40-minute digest intervals
TOP_INTERVALS = 5
DIGEST_TOKEN_BUDGET = 4000           # approx. len(json)/4
SCENARIO_METRICS = ("score", "kwh", "V_deg_person_h", "dev_c", "moves", "switches",
                    "unserved_person_min", "hard_violations")
FORBIDDEN_HINT = ("hard constraints, the comfort band, capacities, validator rules, the structure "
                  "of Equation 4, the seeds, and the prompts of L1-L4")

# ---------------------------------------------------------------------------- schema
_VALUE = {"anyOf": [{"type": "number"}, {"type": "array", "items": {"type": "integer"}}]}
PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "review_id": {"type": "string"},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"interval": {"type": "string"}, "observation": {"type": "string"}},
                "required": ["interval", "observation"],
                "additionalProperties": False,
            },
        },
        "diagnosis": {"type": "string"},
        "diff": {
            "type": "object",
            "properties": {
                "param": {"type": "string", "enum": list(L5_PARAM_SPACE)},
                "from": _VALUE,
                "to": _VALUE,
            },
            "required": ["param", "from", "to"],
            "additionalProperties": False,
        },
        "expected_effect": {
            "type": "object",
            "properties": {
                "metric": {"type": "string"},
                "direction": {"type": "string", "enum": ["increase", "decrease"]},
                "scenarios": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["metric", "direction", "scenarios"],
            "additionalProperties": False,
        },
        "risk": {"type": "string"},
    },
    "required": ["review_id", "evidence", "diagnosis", "diff", "expected_effect", "risk"],
    "additionalProperties": False,
}


# ------------------------------------------------------------------- value helpers
def _norm(v: Any) -> Any:
    """Comparable form of a parameter value. Never raises: a value no declared value can equal
    (a list with a non-integer, a number too large for a float) becomes a unique marker."""
    if isinstance(v, (list, tuple)):
        if all(isinstance(x, int) and not isinstance(x, bool)
               or isinstance(x, float) and math.isfinite(x) and x.is_integer() for x in v):
            return tuple(int(x) for x in v)
        return ("invalid", repr(v))
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        try:
            return round(float(v), 6)
        except OverflowError:
            return ("invalid", repr(v))
    return v


def space_index(param: str, value: Any) -> int | None:
    for k, cand in enumerate(L5_PARAM_SPACE.get(param, [])):
        try:
            if _norm(cand) == _norm(value):
                return k
        except (TypeError, ValueError):
            return None
    return None


def current_values(cfg: PlannerConfig) -> dict[str, Any]:
    return {p: getattr(cfg, p) for p in L5_PARAM_SPACE}


def apply_diff(cfg: PlannerConfig, param: str, to: Any) -> PlannerConfig:
    k = space_index(param, to)
    if k is None:
        raise ValueError(f"{param}={to!r} outside the declared space")
    canonical = L5_PARAM_SPACE[param][k]
    return replace(cfg, **{param: list(canonical) if isinstance(canonical, (list, tuple)) else canonical})


def diff_key(param: Any, frm: Any, to: Any) -> tuple:
    try:
        return (str(param), _norm(frm), _norm(to))
    except (TypeError, ValueError):
        return (str(param), repr(frm), repr(to))


# --------------------------------------------------------------------------- digest
def _interval_id(scenario: str, seed: int, mismatch: bool, start_min: int, end_min: int) -> str:
    sc = f"{scenario}~mm" if mismatch else scenario
    return f"{sc}/s{seed}/{clock_str(start_min)}-{clock_str(end_min)}"


def interval_table(rows: list[dict[str, Any]], window_steps: int = WINDOW_STEPS) -> list[dict[str, Any]]:
    """All 40-minute intervals of all episodes that carry step logs (``_steps``)."""
    w = REFERENCE_WEIGHTS
    out: list[dict[str, Any]] = []
    for r in rows:
        steps = r.get("_steps") or []
        for a in range(0, len(steps), window_steps):
            chunk = steps[a : a + window_steps]
            if not chunk:
                continue
            start = parse_clock(chunk[0]["clock"])
            end = parse_clock(chunk[-1]["clock"]) + STEP_MIN
            E = sum(float(s.get("E", 0.0)) for s in chunk)
            V = sum(float(s.get("V", 0.0)) for s in chunk)
            M = sum(int(s.get("M", 0)) for s in chunk)
            K = sum(int(s.get("K", 0)) for s in chunk)
            preds = [s.get("pred_first_V") for s in chunk]
            pred_V = None if all(p is None for p in preds) else sum(float(p or 0.0) for p in preds)
            src = Counter(str(s.get("source", "unknown")) for s in chunk)
            out.append(
                {
                    "id": _interval_id(r["scenario"], int(r["seed"]), bool(r.get("mismatch")), start, end),
                    "scenario": r["scenario"],
                    "contribution": E + w["lambda_T"] * V + w["lambda_M"] * M + w["lambda_S"] * K,
                    "E": E,
                    "V": V,
                    "M": M,
                    "K": K,
                    "pred_V": pred_V,
                    "realized_V": V,
                    "unserved_person_min": sum(int(s.get("unserved", 0)) for s in chunk) * STEP_MIN,
                    "sources": dict(sorted(src.items())),
                }
            )
    return out


def _round(x: Any, nd: int = 4) -> Any:
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return {k: _round(v, nd) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v, nd) for v in x]
    return x


def scenario_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    g: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        g[r["scenario"]].append(r)
    out = []
    for sc in sorted(g):
        rs = g[sc]
        row: dict[str, Any] = {"scenario": sc, "n": len(rs)}
        for m in SCENARIO_METRICS:
            vals = [float(r[m]) for r in rs if r.get(m) is not None]
            row[m] = round(float(np.mean(vals)), 4) if vals else None
        out.append(row)
    return out


def _history_view(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for h in history:
        t = h.get("tuning") or {}
        out.append(
            _round(
                {
                    "review_id": h.get("review_id"),
                    "param": h.get("param"),
                    "from": h.get("from"),
                    "to": h.get("to"),
                    "accepted": bool(h.get("accepted")),
                    "stage": h.get("stage"),
                    "mean_score_diff": t.get("mean_diff"),
                    "ci": [t.get("lo"), t.get("hi")] if t else None,
                }
            )
        )
    return out


def build_digest(
    rows: list[dict[str, Any]],
    cfg: PlannerConfig,
    history: list[dict[str, Any]] | None = None,
    decisions_rejected: Iterable[dict[str, Any]] = (),
    top_k: int = TOP_INTERVALS,
) -> dict[str, Any]:
    """Deterministic digest of one development sweep (≈4,000 tokens at most).

    ``rows`` are episode summaries; rows that carry ``_steps`` contribute intervals; rows that
    carry ``_decisions`` (or ``decisions_rejected``) contribute validator rejection reasons.
    """
    history = history or []
    ivs = interval_table(rows)
    ivs.sort(key=lambda r: (-r["contribution"], r["id"]))
    top = ivs[:top_k]
    top_ids = {r["id"] for r in top}
    worst_per_scenario = []
    seen_sc = set()
    for r in ivs:
        if r["scenario"] not in seen_sc:
            seen_sc.add(r["scenario"])
            if r["id"] not in top_ids:
                worst_per_scenario.append(r)
    worst_per_scenario.sort(key=lambda r: r["id"])
    codes: Counter[str] = Counter()
    for r in rows:
        for d in r.get("_decisions") or []:
            for rej in d.get("rejected_sample") or []:
                for v in rej.get("violations", []):
                    codes[f"{rej.get('source', '?')}:{v.get('code', '?')}"] += 1
    for rej in decisions_rejected:
        for v in rej.get("violations", []):
            codes[f"{rej.get('source', '?')}:{v.get('code', '?')}"] += 1
    src_total: Counter[str] = Counter()
    for r in ivs:
        src_total.update(r["sources"])

    def iv_view(r: dict[str, Any]) -> dict[str, Any]:
        return _round({k: r[k] for k in ("id", "contribution", "E", "V", "M", "K", "pred_V", "realized_V",
                                         "unserved_person_min", "sources")})

    digest: dict[str, Any] = {
        "config": {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in current_values(cfg).items()},
        "param_space": L5_PARAM_SPACE,
        "reference_weights": REFERENCE_WEIGHTS,
        "scenarios": scenario_summary(rows),
        "top_intervals": [iv_view(r) for r in top],
        "worst_interval_per_scenario": [iv_view(r) for r in worst_per_scenario],
        "committed_action_sources": dict(sorted(src_total.items())),
        "rejected_plan_reasons": dict(sorted(codes.most_common(10))),
        "history": _history_view(history),
    }
    # keep within the token budget: drop oldest history first, then per-scenario intervals
    while len(json.dumps(digest)) / 4 > DIGEST_TOKEN_BUDGET and digest["history"]:
        digest["history"] = digest["history"][1:]
    while len(json.dumps(digest)) / 4 > DIGEST_TOKEN_BUDGET and digest["worst_interval_per_scenario"]:
        digest["worst_interval_per_scenario"] = digest["worst_interval_per_scenario"][:-1]
    digest["interval_ids"] = sorted(
        [r["id"] for r in digest["top_intervals"]] + [r["id"] for r in digest["worst_interval_per_scenario"]]
    )
    return digest


# ------------------------------------------------------------------ evidence checker
@dataclass(frozen=True)
class ProposalIssue:
    code: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "detail": self.detail}


def check_proposal(
    p: Any,
    digest: dict[str, Any] | None,
    cfg: PlannerConfig,
    history: list[dict[str, Any]] | None = None,
    require_evidence: bool = True,
) -> list[ProposalIssue]:
    """Schema check against Table 9 and the evidence checker. Runs before any sweep is spent."""
    issues: list[ProposalIssue] = []
    if not isinstance(p, dict):
        return [ProposalIssue("schema", "proposal is not an object")]
    diff = p.get("diff")
    if not isinstance(diff, dict) or not {"param", "from", "to"} <= set(diff):
        return [ProposalIssue("schema", "diff must have param, from, to")]
    param, frm, to = diff["param"], diff["from"], diff["to"]
    if param not in L5_PARAM_SPACE:
        return [ProposalIssue("forbidden_param", f"{param!r} is not in the declared diff space (Table 9)")]
    cur = getattr(cfg, param)
    if _norm(frm) != _norm(cur):
        issues.append(ProposalIssue("wrong_from", f"from={frm!r} differs from current {param}={cur!r}"))
    i_from, i_to = space_index(param, frm), space_index(param, to)
    if i_to is None:
        issues.append(ProposalIssue("out_of_range", f"to={to!r} outside {L5_PARAM_SPACE[param]}"))
    elif i_from is None:
        issues.append(ProposalIssue("out_of_range", f"from={frm!r} outside {L5_PARAM_SPACE[param]}"))
    elif abs(i_to - i_from) != 1:
        issues.append(ProposalIssue("not_one_step", f"{frm!r} -> {to!r} is not one step along {L5_PARAM_SPACE[param]}"))
    if require_evidence:
        ev = p.get("evidence")
        if not isinstance(ev, list) or not ev:
            issues.append(ProposalIssue("no_evidence", "at least one cited interval is required"))
        else:
            known = set((digest or {}).get("interval_ids", []))
            for e in ev:
                if not isinstance(e, dict):
                    issues.append(ProposalIssue("schema", "evidence items must be objects"))
                    continue
                iv = str(e.get("interval", ""))
                if iv not in known:
                    issues.append(ProposalIssue("unknown_interval", f"interval {iv!r} is not in the digest"))
                if re.search(r"\d", str(e.get("observation", ""))):
                    issues.append(ProposalIssue("number_in_observation",
                                                "observations may not contain numbers; cite the interval id instead"))
    rejected = {
        diff_key(h.get("param"), h.get("from"), h.get("to"))
        for h in (history or [])
        if h.get("valid", True) and not h.get("accepted") and h.get("param") is not None
    }
    if diff_key(param, frm, to) in rejected:
        issues.append(ProposalIssue("repeat_rejected", f"{param} {frm!r}->{to!r} was already rejected"))
    return issues


# ------------------------------------------------------------------- sweeps and rule
SweepFn = Callable[[PlannerConfig, tuple[int, ...], tuple[str, ...], bool], list[dict[str, Any]]]


def sweep(
    cfg: PlannerConfig,
    seeds: Iterable[int],
    scenarios: Iterable[str] = ALL_SCENARIOS,
    keep_steps: bool = False,
    workers: int | None = None,
    policy: str = "arm_B",
) -> list[dict[str, Any]]:
    """One development sweep of the non-LLM evaluation arm (masks + rule-staged plans)."""
    from ..experiments.common import episode_job, run_parallel

    specs = [
        {"cfg": cfg.to_dict(), "scenario": sc, "seed": int(s), "policy": policy, "keep_steps": keep_steps}
        for sc in scenarios
        for s in seeds
    ]
    return run_parallel(episode_job, specs, workers)


def tuning_test(base: list[dict[str, Any]], cand: list[dict[str, Any]], n_boot: int = 10_000, seed: int = 0) -> dict[str, Any]:
    """Criteria (i) and (ii) on tuning seeds."""
    key = lambda r: (r["scenario"], int(r["seed"]), bool(r.get("mismatch", False)))
    b = {key(r): r for r in base}
    c = {key(r): r for r in cand}
    keys = sorted(set(b) & set(c))
    if not keys:
        return {"pass": False, "reason": "no paired episodes"}
    bs = paired_bootstrap([float(c[k]["score"]) for k in keys], [float(b[k]["score"]) for k in keys],
                          n_boot=n_boot, seed=seed)
    improves = bs["mean_diff"] < 0 and bs["hi"] < 0
    per_sc: dict[str, dict[str, Any]] = {}
    ok_safety = True
    for sc in sorted({k[0] for k in keys}):
        ks = [k for k in keys if k[0] == sc]
        ub = sum(float(b[k].get("unserved_person_min", 0)) for k in ks)
        uc = sum(float(c[k].get("unserved_person_min", 0)) for k in ks)
        hb = sum(float(b[k].get("hard_violations", 0)) for k in ks)
        hc = sum(float(c[k].get("hard_violations", 0)) for k in ks)
        ok = uc <= ub + 1e-9 and hc <= hb + 1e-9
        ok_safety &= ok
        per_sc[sc] = {"unserved_base": ub, "unserved_cand": uc, "violations_base": hb, "violations_cand": hc, "ok": ok}
    reason = "pass" if (improves and ok_safety) else ("no significant improvement" if not improves else "unserved or violations increased")
    return {"pass": bool(improves and ok_safety), "reason": reason, "bootstrap": bs, "per_scenario": per_sc}


def confirmation_test(base_check: list[dict[str, Any]], cand_check: list[dict[str, Any]]) -> dict[str, Any]:
    """Criterion (iii): the confirmation sweep on check seeds does not worsen the mean score."""
    mb = float(np.mean([r["score"] for r in base_check])) if base_check else float("nan")
    mc = float(np.mean([r["score"] for r in cand_check])) if cand_check else float("nan")
    ok = bool(base_check and cand_check and mc <= mb + 1e-9)
    return {"pass": ok, "base_mean": mb, "cand_mean": mc, "reason": "pass" if ok else "worse on check seeds"}


def acceptance_rule(
    base_tune: list[dict[str, Any]],
    cand_tune: list[dict[str, Any]],
    base_check: list[dict[str, Any]] | None = None,
    cand_check: list[dict[str, Any]] | None = None,
    n_boot: int = 10_000,
) -> dict[str, Any]:
    t = tuning_test(base_tune, cand_tune, n_boot=n_boot)
    out: dict[str, Any] = {"tuning": t, "accepted": False, "stage": "tuning"}
    if not t["pass"]:
        return out
    if base_check is None or cand_check is None:
        out["stage"] = "confirmation_pending"
        return out
    cf = confirmation_test(base_check, cand_check)
    out["confirmation"] = cf
    out["stage"] = "accepted" if cf["pass"] else "confirmation"
    out["accepted"] = cf["pass"]
    return out


# ------------------------------------------------------------------------ proposers
@dataclass
class ProposalResult:
    proposal: dict[str, Any] | None
    transcript: dict[str, Any] | None = None
    error: str | None = None


class Proposer(Protocol):
    name: str
    require_evidence: bool

    def propose(self, digest: dict[str, Any], cfg: PlannerConfig, history: list[dict[str, Any]],
                review_id: str) -> ProposalResult: ...


class RandomSearchProposer:
    """Control: single-parameter one-step diffs drawn uniformly from Table 9.

    A parameter is drawn uniformly among those with an admissible neighbour, then a neighbour
    uniformly; diffs already rejected are excluded (the L5 checker forbids repeats too).
    """

    name = "random"
    require_evidence = False

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def candidates(self, cfg: PlannerConfig, history: list[dict[str, Any]]) -> dict[str, list[Any]]:
        rejected = {diff_key(h.get("param"), h.get("from"), h.get("to"))
                    for h in history if h.get("valid", True) and not h.get("accepted")}
        out: dict[str, list[Any]] = {}
        for p, space in L5_PARAM_SPACE.items():
            k = space_index(p, getattr(cfg, p))
            if k is None:
                continue
            nb = [space[j] for j in (k - 1, k + 1) if 0 <= j < len(space)]
            nb = [v for v in nb if diff_key(p, getattr(cfg, p), v) not in rejected]
            if nb:
                out[p] = nb
        return out

    def propose(self, digest: dict[str, Any], cfg: PlannerConfig, history: list[dict[str, Any]],
                review_id: str) -> ProposalResult:
        cands = self.candidates(cfg, history)
        if not cands:
            return ProposalResult(None, error="no admissible diff left")
        p = self.rng.choice(sorted(cands))
        to = self.rng.choice(cands[p])
        frm = getattr(cfg, p)
        prop = {
            "review_id": review_id,
            "evidence": [],
            "diagnosis": "random-search control",
            "diff": {"param": p, "from": list(frm) if isinstance(frm, (list, tuple)) else frm,
                     "to": list(to) if isinstance(to, (list, tuple)) else to},
            "expected_effect": {"metric": "score", "direction": "decrease", "scenarios": []},
            "risk": "none stated (control)",
        }
        return ProposalResult(prop)


L5_SYSTEM = f"""You are the episode reviewer (L5) of CampusFlex, a simulation-checked controller for six \
campus study rooms. You act between runs, never inside one. You receive a deterministic digest of a \
development sweep and propose exactly ONE bounded configuration change.

Rules (proposals that break them are rejected before any evaluation is spent):
- Change exactly one parameter from "param_space", by exactly one step along its ordered list. \
  "diff.from" must equal the current value in "config".
- You may not change {FORBIDDEN_HINT}.
- Cite evidence as interval identifiers that appear in the digest ("top_intervals" or \
  "worst_interval_per_scenario"). The "observation" text must not contain any digits: every number you \
  would cite is already in the digest under the interval identifier.
- Do not repeat a diff that the "history" shows as rejected; learn from measured outcomes there.
- The diff will be accepted only if it improves the mean full-evening score (reference weights) on tuning \
  seeds with a paired bootstrap interval excluding zero, does not increase unserved person-minutes or \
  hard-constraint violations in any scenario, and does not worsen held-back check seeds.
Return a single JSON object with review_id, evidence, diagnosis, diff, expected_effect and risk."""


class LLMReviewer:
    """L5 proposer backed by an LLM client (role "L5")."""

    name = "L5"
    require_evidence = True

    def __init__(self, client: Any, budget: Any | None = None):
        self.client = client
        self.budget = budget

    @staticmethod
    def build_prompt(digest: dict[str, Any], review_id: str) -> str:
        view = {k: v for k, v in digest.items() if k != "interval_ids"}
        return (f"review_id to use: {review_id}\nDigest of the latest development sweep (JSON):\n"
                + json.dumps(view, indent=1, sort_keys=True))

    def propose(self, digest: dict[str, Any], cfg: PlannerConfig, history: list[dict[str, Any]],
                review_id: str) -> ProposalResult:
        from ..llm.client import call

        r = call(self.client, self.budget, role="L5", system=L5_SYSTEM, user=self.build_prompt(digest, review_id),
                 schema=PROPOSAL_SCHEMA)
        tr = r.transcript()
        if not r.ok or not isinstance(r.data, dict):
            return ProposalResult(None, tr, r.error or "no data")
        prop = dict(r.data)
        prop.setdefault("review_id", review_id)
        return ProposalResult(prop, tr)


# -------------------------------------------------------------------------- the loop
@dataclass
class ReviewOutcome:
    label: str
    initial_cfg: dict[str, Any]
    final_cfg: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)
    rounds: int = 0
    proposals: int = 0
    sweeps: dict[str, int] = field(default_factory=lambda: {"baseline": 0, "tuning": 0, "confirmation": 0})
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> list[dict[str, Any]]:
        return [h for h in self.history if h.get("accepted")]

    @property
    def acceptance_rate(self) -> float | None:
        evaluated = [h for h in self.history if h.get("valid")]
        return (len(self.accepted) / len(evaluated)) if evaluated else None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["accepted"] = self.accepted
        d["review_acceptance_rate"] = self.acceptance_rate
        return d


class ReviewLoop:
    """At most ``max_rounds`` rounds of up to ``per_round`` proposals (6 × 3 = 18 by default);
    stops early after ``stop_after_empty_rounds`` consecutive rounds with no accepted diff."""

    def __init__(
        self,
        proposer: Proposer,
        cfg: PlannerConfig,
        scenarios: Iterable[str] = ALL_SCENARIOS,
        tuning_seeds: Iterable[int] = TUNING_SEEDS,
        check_seeds: Iterable[int] = CHECK_SEEDS,
        max_rounds: int = 6,
        per_round: int = 3,
        stop_after_empty_rounds: int | None = 2,
        sweep_fn: SweepFn | None = None,
        workers: int | None = None,
        out_config: str | Path | None = None,
        changelog: str | Path | None = None,
        label: str | None = None,
        n_boot: int = 10_000,
    ):
        self.proposer = proposer
        self.cfg = cfg
        self.scenarios = tuple(scenarios)
        self.tuning_seeds = tuple(tuning_seeds)
        self.check_seeds = tuple(check_seeds)
        self.max_rounds = max_rounds
        self.per_round = per_round
        self.stop_after = stop_after_empty_rounds
        self.workers = workers
        self.sweep_fn: SweepFn = sweep_fn or (
            lambda c, seeds, scs, keep: sweep(c, seeds, scs, keep_steps=keep, workers=self.workers)
        )
        self.label = label or proposer.name
        self.out_config = Path(out_config) if out_config else None
        self.changelog = Path(changelog) if changelog else None
        self.n_boot = n_boot
        self._cache: dict[tuple, list[dict[str, Any]]] = {}

    def _run(self, cfg: PlannerConfig, seeds: tuple[int, ...], keep: bool, kind: str, out: ReviewOutcome) -> list[dict[str, Any]]:
        k = (json.dumps(cfg.to_dict(), sort_keys=True), seeds, self.scenarios, keep)
        if k not in self._cache:
            # a cached sweep with step logs also serves requests without them
            k2 = (k[0], seeds, self.scenarios, True)
            if not keep and k2 in self._cache:
                return self._cache[k2]
            self._cache[k] = self.sweep_fn(cfg, seeds, self.scenarios, keep)
            out.sweeps[kind] += 1
        return self._cache[k]

    def _log_changelog(self, rec: dict[str, Any]) -> None:
        if self.changelog is None:
            return
        self.changelog.parent.mkdir(parents=True, exist_ok=True)
        log = json.loads(self.changelog.read_text()) if self.changelog.exists() else []
        t = rec.get("tuning", {}).get("bootstrap", {})
        log.append({
            "review_id": rec["review_id"],
            "commit_message": f"{rec['review_id']}: {rec['param']} {rec['from']} -> {rec['to']} ({self.label})",
            "param": rec["param"],
            "from": rec["from"],
            "to": rec["to"],
            "mean_score_diff": t.get("mean_diff"),
            "ci": [t.get("lo"), t.get("hi")],
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        self.changelog.write_text(json.dumps(log, indent=1, default=str))

    def run(self) -> ReviewOutcome:
        cfg = self.cfg
        out = ReviewOutcome(label=self.label, initial_cfg=cfg.to_dict(), final_cfg=cfg.to_dict(),
                            meta={"scenarios": list(self.scenarios), "tuning_seeds": list(self.tuning_seeds),
                                  "check_seeds": list(self.check_seeds), "max_rounds": self.max_rounds,
                                  "per_round": self.per_round})
        history = out.history
        empty_rounds = 0
        for rnd in range(1, self.max_rounds + 1):
            out.rounds = rnd
            accepted_this_round = False
            for j in range(1, self.per_round + 1):
                review_id = f"R{rnd}-p{j}"
                base_tune = self._run(cfg, self.tuning_seeds, True, "baseline", out)
                digest = build_digest(base_tune, cfg, history)
                res = self.proposer.propose(digest, cfg, history, review_id)
                out.proposals += 1
                rec: dict[str, Any] = {"review_id": review_id, "round": rnd, "proposer": self.label,
                                       "proposal": res.proposal, "transcript": res.transcript}
                if res.proposal is None:
                    rec.update({"valid": False, "accepted": False, "stage": "no_proposal",
                                "violations": [{"code": "no_proposal", "detail": res.error or ""}]})
                    history.append(rec)
                    continue
                d = res.proposal.get("diff") if isinstance(res.proposal, dict) else None
                if isinstance(d, dict):
                    rec.update({"param": d.get("param"), "from": d.get("from"), "to": d.get("to")})
                issues = check_proposal(res.proposal, digest, cfg, history,
                                        require_evidence=getattr(self.proposer, "require_evidence", True))
                if issues:
                    rec.update({"valid": False, "accepted": False, "stage": "evidence",
                                "violations": [i.to_dict() for i in issues]})
                    history.append(rec)
                    continue
                cand = apply_diff(cfg, d["param"], d["to"])
                cand_tune = self._run(cand, self.tuning_seeds, True, "tuning", out)
                t = tuning_test(base_tune, cand_tune, n_boot=self.n_boot)
                rec.update({"valid": True, "tuning": {**t["bootstrap"], "reason": t["reason"]} if "bootstrap" in t else t,
                            "per_scenario": t.get("per_scenario")})
                if not t["pass"]:
                    rec.update({"accepted": False, "stage": "tuning"})
                    history.append(rec)
                    continue
                base_check = self._run(cfg, self.check_seeds, False, "baseline", out)
                cand_check = self._run(cand, self.check_seeds, False, "confirmation", out)
                cf = confirmation_test(base_check, cand_check)
                rec["confirmation"] = cf
                if not cf["pass"]:
                    rec.update({"accepted": False, "stage": "confirmation"})
                    history.append(rec)
                    continue
                rec.update({"accepted": True, "stage": "accepted"})
                history.append(rec)
                cfg = cand
                accepted_this_round = True
                if self.out_config is not None:
                    cfg.save(self.out_config)
                self._log_changelog({**rec, "tuning": {"bootstrap": t["bootstrap"]}})
            empty_rounds = 0 if accepted_this_round else empty_rounds + 1
            if self.stop_after is not None and empty_rounds >= self.stop_after:
                break
        out.final_cfg = cfg.to_dict()
        if self.out_config is not None and not out.accepted:
            cfg.save(self.out_config)          # unchanged configuration is still frozen for E6
        return out


# ------------------------------------------------------------------------- report
def _fmt(x: Any, nd: int = 3) -> str:
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(_fmt(v, nd) for v in x) + "]"
    return str(x)


def report_values(outcome: ReviewOutcome) -> dict[str, Any]:
    """The values a review report may cite: sweep counts and, per accepted diff, the declared
    parameter values and the measured tuning/confirmation statistics. Deliberately excludes the
    proposer's own output (proposal text, transcript, token counts, latency), so a number the
    proposer wrote cannot verify itself."""
    return {
        "counts": {"rounds": outcome.rounds, "proposals": outcome.proposals,
                   "accepted": len(outcome.accepted), "sweeps": dict(outcome.sweeps)},
        "accepted": [
            {
                "from": h.get("from"),
                "to": h.get("to"),
                "tuning": {k: (h.get("tuning") or {}).get(k) for k in ("mean_diff", "lo", "hi", "n")},
                "confirmation": {k: (h.get("confirmation") or {}).get(k) for k in ("base_mean", "cand_mean")},
            }
            for h in outcome.accepted
        ],
    }


def write_report(outcome: ReviewOutcome, path: str | Path | None = None) -> dict[str, Any]:
    """Short report for people. Numbers are copied from the outcome record and verified by the
    faithfulness checker used for L4; any number that does not match a logged value is removed."""
    path = Path(path) if path else ROOT / "docs" / "l5_report.md"
    lines = [f"# L5 review report ({outcome.label})", ""]
    lines.append(f"Rounds run: {outcome.rounds}. Proposals made: {outcome.proposals}. "
                 f"Accepted diffs: {len(outcome.accepted)}.")
    lines.append(f"Sweeps spent: tuning {outcome.sweeps['tuning']}, confirmation {outcome.sweeps['confirmation']}, "
                 f"baseline {outcome.sweeps['baseline']}.")
    lines.append("")
    if not outcome.accepted:
        lines.append("No diff was accepted; the configuration is unchanged.")
    for h in outcome.accepted:
        t = h.get("tuning") or {}
        cf = h.get("confirmation") or {}
        prop = h.get("proposal") or {}
        ivs = ", ".join(f"`{e.get('interval')}`" for e in prop.get("evidence", []) if isinstance(e, dict)) or "none cited"
        lines += [
            f"## {h['review_id']}: `{h['param']}` {_fmt(h['from'])} -> {_fmt(h['to'])}",
            "",
            f"- Motivating intervals: {ivs}",
            f"- Diagnosis (proposer text): {prop.get('diagnosis', '')}",
            f"- Measured effect on tuning seeds: mean score change {_fmt(t.get('mean_diff'))} "
            f"(95% interval {_fmt(t.get('lo'))} to {_fmt(t.get('hi'))}, n = {t.get('n')}).",
            f"- Confirmation seeds: mean score {_fmt(cf.get('base_mean'))} before, {_fmt(cf.get('cand_mean'))} after.",
            "",
        ]
    rejected = [h for h in outcome.history if not h.get("accepted")]
    if rejected:
        lines += ["## Rejected or invalid proposals", "", "| review | param | stage | reason |", "|---|---|---|---|"]
        for h in rejected:
            reason = (h.get("tuning") or {}).get("reason") or ", ".join(v.get("code", "") for v in h.get("violations", []))
            lines.append(f"| {h['review_id']} | {h.get('param', '')} | {h.get('stage', '')} | {reason} |")
    text = "\n".join(lines) + "\n"
    fr = faithfulness_check(text, report_values(outcome), extra_allowed=[95.0])
    out = fr.cleaned_text
    model = (outcome.meta or {}).get("model")
    if model:      # metadata, added after the number check (a model id may contain digits)
        out = out.replace("\n\n", f"\n\nProposer model: `{model}`.\n\n", 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(out)
    return {"path": str(path), "faithfulness": fr.to_dict()}
