"""Policies: v1 baselines, the v2 efficiency-ranked threshold, and the simulation planner
with its E1 arms (A masks; B +rule-staged; C +LLM-staged; D +both).

All policies return a ``Decision``; only the episode runner / commit node calls
``apply_plan``. The planner ranks every surviving candidate with Equation 4.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Protocol

import numpy as np

from .candidates import mask_candidates, rule_staged_macros
from .config import ROOM_IDS, PlannerConfig
from .forecast import ConstantForecaster, Forecast
from .model import EFFICIENCY_ORDER, INVENTORY_ORDER
from .plans import CompiledPlan, StepAction, compile_macro, mask_action
from .simulate import RolloutResult, rollout
from .state import Snapshot
from .validator import validate_action, validate_macro, validate_plan


@dataclass
class Decision:
    action: StepAction
    plan: CompiledPlan | None = None
    predicted: RolloutResult | None = None
    ranking: list[dict[str, Any]] = field(default_factory=list)
    n_candidates: dict[str, int] = field(default_factory=dict)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    trigger: str | None = None
    llm: list[dict[str, Any]] = field(default_factory=list)
    forecast: dict[str, Any] | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def plan_id(self) -> str:
        return self.plan.plan_id if self.plan else "baseline"

    @property
    def source(self) -> str:
        return self.plan.source if self.plan else "baseline"


class Policy(Protocol):
    name: str

    def reset(self) -> None: ...
    def decide(self, snap: Snapshot) -> Decision: ...


def hold_action(snap: Snapshot, cfg: PlannerConfig, fill: str = "smart") -> StepAction:
    a = mask_action(snap.open_set, snap.health, cfg)
    return replace(a, fill=fill)


def _respect_dwell(desired: np.ndarray, snap: Snapshot, cfg: PlannerConfig) -> np.ndarray:
    out = desired.copy()
    for i in range(len(ROOM_IDS)):
        if out[i] != snap.open[i] and snap.steps_since_switch[i] < cfg.min_room_dwell_steps:
            out[i] = snap.open[i]
    return out


# ----------------------------------------------------------------------------- baselines
class FixedOpening:
    """Open the inventory-order rooms that cover the planned seats all evening."""

    def __init__(self, cfg: PlannerConfig):
        self.cfg = cfg
        self.name = "fixed"

    def reset(self) -> None:
        pass

    def decide(self, snap: Snapshot) -> Decision:
        mask, cap = set(), 0
        for i in INVENTORY_ORDER:
            if cap >= self.cfg.fixed_planned_seats:
                break
            if snap.capacity[i] > 0:
                mask.add(ROOM_IDS[i])
                cap += int(snap.capacity[i])
        a = replace(mask_action(frozenset(mask), snap.health, self.cfg), fill="inventory")
        return Decision(action=a)


class DemandThreshold:
    """Open rooms in a fixed order until open capacity covers current demand.

    ``order="inventory"`` is the v1 rule; ``order="efficiency"`` is the v2 control that
    opens rooms by predicted kWh per seat-hour (removes the advantage of room ordering alone).
    """

    def __init__(self, cfg: PlannerConfig, order: str = "inventory"):
        self.cfg = cfg
        self.order = INVENTORY_ORDER if order == "inventory" else EFFICIENCY_ORDER
        self.fill = order
        self.name = "threshold" if order == "inventory" else "threshold_eff"

    def reset(self) -> None:
        pass

    def decide(self, snap: Snapshot) -> Decision:
        want = (snap.counts > 0) & (snap.capacity > 0)
        cap = int(np.sum(snap.capacity[want]))
        for i in self.order:
            if cap >= snap.demand:
                break
            if not want[i] and snap.capacity[i] > 0:
                want[i] = True
                cap += int(snap.capacity[i])
        # close empty rooms that are no longer needed (reverse order)
        for i in reversed(self.order):
            if want[i] and snap.counts[i] == 0 and cap - snap.capacity[i] >= snap.demand:
                want[i] = False
                cap -= int(snap.capacity[i])
        want = _respect_dwell(want, snap, self.cfg)
        mask = frozenset(r for r, w in zip(ROOM_IDS, want) if w)
        a = replace(mask_action(mask, snap.health, self.cfg), fill=self.fill)
        return Decision(action=a)


# ----------------------------------------------------------------------- planner + arms
@dataclass
class ProposalBatch:
    macros: list[Any] = field(default_factory=list)       # raw dicts / JSON strings from the proposer
    transcript: dict[str, Any] | None = None
    error: str | None = None


class StagedProposer(Protocol):
    """L2 interface: propose up to N macro-plans from a state summary."""

    name: str

    def propose(self, snap: Snapshot, fc: Forecast, feedback: list[dict[str, Any]]) -> ProposalBatch: ...


Negotiator = Callable[[Snapshot, list[tuple[CompiledPlan, RolloutResult]]], dict[str, Any] | None]


class SimulationPlanner:
    """Simulation-checked replanning: rank candidates by Equation 4 and commit the first step."""

    def __init__(
        self,
        cfg: PlannerConfig,
        forecaster: Any | None = None,
        use_masks: bool = True,
        use_rule: bool = False,
        l2: StagedProposer | None = None,
        negotiator: Negotiator | None = None,
        name: str | None = None,
        keep_ranking: int = 5,
    ):
        self.cfg = cfg
        self.forecaster = forecaster or ConstantForecaster()
        self.use_masks = use_masks
        self.use_rule = use_rule
        self.l2 = l2
        self.negotiator = negotiator
        self.keep_ranking = keep_ranking
        self.name = name or ("planner" + ("+rule" if use_rule else "") + ("+L2" if l2 else ""))
        self.reset()

    def reset(self) -> None:
        self.carry: CompiledPlan | None = None
        self.prev_health: np.ndarray | None = None
        self.prev_demand: int | None = None
        self.l2_feedback: list[dict[str, Any]] = []
        # l2_fallbacks: the call failed (error, timeout, unparseable); l2_empty: a valid reply with no
        # plans ("nothing staged would help" – the prompt allows it), which is an LLM answer, not a fallback
        self.stats = {"l2_calls": 0, "l2_plans": 0, "l2_valid": 0, "l2_selected": 0, "l2_fallbacks": 0,
                      "l2_empty": 0}
        if hasattr(self.forecaster, "reset"):
            self.forecaster.reset()

    # -------------------------------------------------------------- triggers
    def l2_trigger(self, snap: Snapshot, fc: Forecast, hold: RolloutResult | None) -> str | None:
        reasons = []
        if self.prev_health is not None and np.any((snap.health <= 0) & (self.prev_health > 0)):
            reasons.append("fault")
        elif np.any((snap.health <= 0) & (snap.counts > 0)):
            reasons.append("fault")
        if hold is not None and hold.temps is not None:
            occ = (hold.counts > 0).any(axis=0) | (snap.counts > 0)
            if np.any(hold.temps[1:, occ] > self.cfg.l2_temp_trigger_c):
                reasons.append("band")
        if self.prev_demand is not None and abs(snap.demand - self.prev_demand) > self.cfg.l2_demand_trigger:
            reasons.append("demand")
        if len(fc.demand) and abs(int(max(fc.demand, key=lambda x: abs(x - snap.demand))) - snap.demand) > self.cfg.l2_demand_trigger:
            reasons.append("forecast")
        if snap.step % max(1, self.cfg.l2_periodic_every) == 0:
            reasons.append("periodic")
        return ",".join(reasons) or None

    # ------------------------------------------------------------- decision
    def _compile(self, raw: Any, snap: Snapshot, source: str) -> tuple[CompiledPlan | None, list[dict[str, Any]]]:
        m, viol = validate_macro(raw, snap, self.cfg)
        if m is None or viol:
            pid = m.plan_id if m is not None else (raw.get("plan_id") if isinstance(raw, dict) else "?")
            return None, [{"plan_id": pid, "source": source, "violations": [v.to_dict() for v in viol]}]
        plan = compile_macro(m, snap.open_set, snap.health, self.cfg, self.cfg.horizon_steps, source, snap.step)
        pv = validate_plan(plan, snap, self.cfg)
        if pv:
            return None, [{"plan_id": plan.plan_id, "source": source, "violations": [v.to_dict() for v in pv]}]
        return plan, []

    # Stage methods – one implementation shared by decide() and the LangGraph nodes.
    def stage_forecast(self, snap: Snapshot) -> "PlanContext":
        cfg = self.cfg
        fc = self.forecaster.forecast(snap, cfg.horizon_steps + cfg.omega_steps)
        return PlanContext(snap=snap, fc=fc)

    def stage_pool(self, ctx: "PlanContext") -> "PlanContext":
        """Candidate pool: 64 masks + rule-staged + carried-forward plan, each validated."""
        snap, cfg, fc = ctx.snap, self.cfg, ctx.fc
        if self.use_masks:
            for p in mask_candidates(snap, cfg):
                v = validate_action(p.first(), snap, cfg)
                if v:
                    ctx.rejected.append({"plan_id": p.plan_id, "source": "mask", "violations": [x.to_dict() for x in v]})
                else:
                    ctx.pool.append(p)
            ctx.counts["mask"] = 64
        if self.use_rule:
            macros = rule_staged_macros(snap, fc, cfg)
            ctx.counts["rule"] = len(macros)
            for raw in macros:
                plan, rej = self._compile(raw, snap, "rule")
                ctx.rejected += rej
                if plan:
                    ctx.pool.append(plan)
        if self.carry is not None:
            cp = self.carry.shifted(1)
            self.carry = None
            if cp is not None:
                ctx.counts["carry"] = 1
                v = validate_plan(cp, snap, cfg)
                if v:
                    ctx.rejected.append({"plan_id": cp.plan_id, "source": cp.source, "violations": [x.to_dict() for x in v]})
                else:
                    ctx.pool.append(cp)
        ctx.hold_plan = next((p for p in ctx.pool if p.source == "mask" and p.first().open == snap.open_set), None)
        ctx.hold_r = rollout(snap, ctx.hold_plan, fc, cfg, keep_traj=True) if ctx.hold_plan else None
        if self.l2 is not None:
            ctx.trigger = self.l2_trigger(snap, fc, ctx.hold_r)
        return ctx

    def stage_l2(self, ctx: "PlanContext") -> "PlanContext":
        """LLM node: proposals only. Validation/compilation happens in stage_compile."""
        if self.l2 is None or not ctx.trigger:
            return ctx
        self.stats["l2_calls"] += 1
        t0 = time.monotonic()
        batch = self.l2.propose(ctx.snap, ctx.fc, self.l2_feedback)
        self.l2_feedback = []
        if batch.transcript:
            ctx.llm.append(batch.transcript)
        if batch.error:
            self.stats["l2_fallbacks"] += 1
            ctx.llm.append({"role": "L2", "fallback": batch.error, "latency_s": round(time.monotonic() - t0, 3)})
        elif not batch.macros:
            self.stats["l2_empty"] += 1           # masks + rule plans only at this step, by the LLM's choice
        ctx.l2_raw = list(batch.macros[: self.cfg.l2_max_plans])
        ctx.counts["L2"] = len(ctx.l2_raw)
        return ctx

    def stage_compile(self, ctx: "PlanContext") -> "PlanContext":
        for raw in ctx.l2_raw:
            self.stats["l2_plans"] += 1
            plan, rej = self._compile(raw, ctx.snap, "L2")
            if plan:
                self.stats["l2_valid"] += 1
                plan.plan_id = f"L2:{plan.plan_id}"
                ctx.pool.append(plan)
            else:
                ctx.rejected += rej
                self.l2_feedback += rej       # reason returned to the LLM on its next call
        ctx.l2_raw = []
        return ctx

    def stage_score(self, ctx: "PlanContext") -> "PlanContext":
        snap, cfg, fc = ctx.snap, self.cfg, ctx.fc
        for p in ctx.pool:
            r = ctx.hold_r if (p is ctx.hold_plan and ctx.hold_r is not None) else rollout(snap, p, fc, cfg)
            if r.power_cap_violated:
                ctx.rejected.append({"plan_id": p.plan_id, "source": p.source,
                                     "violations": [{"code": "power_cap", "detail": f"{r.max_power_kw:.2f} kW"}]})
                continue
            if r.later_violations:
                # A later step the validator would refuse when it comes due: the rollout scored
                # trimmed moves the plan cannot realize, so the plan is rejected, not trimmed.
                rej = {"plan_id": p.plan_id, "source": p.source,
                       "violations": [x.to_dict() for x in dict.fromkeys(r.later_violations)]}
                ctx.rejected.append(rej)
                if p.source == "L2":
                    self.l2_feedback.append(rej)
                continue
            ctx.scored.append((p, r))
        # Eq. 4 order (U first, then J); an L1 energy budget ranks compliant plans before
        # non-compliant ones at equal service.
        ctx.scored.sort(key=lambda pr: (pr[1].score.U, pr[1].budget_violated, pr[1].score.J))
        return ctx

    def stage_tradeoff(self, ctx: "PlanContext") -> "PlanContext":
        if self.negotiator is not None and ctx.scored:
            out = self.negotiator(ctx.snap, ctx.scored)
            if out:
                ctx.notes["negotiation"] = {k: v for k, v in out.items() if k != "transcript"}
                ctx.chosen_idx = int(out.get("chosen_index", 0))
                if out.get("transcript"):
                    ctx.llm.append(out["transcript"])
        return ctx

    def stage_decision(self, ctx: "PlanContext") -> Decision:
        snap = ctx.snap
        if not ctx.scored:
            return Decision(action=hold_action(snap, self.cfg), rejected=ctx.rejected, trigger=ctx.trigger,
                            llm=ctx.llm, notes={"fallback": "no feasible candidate; hold"})
        plan, res = ctx.scored[ctx.chosen_idx]
        if plan.source in ("L2", "carry:L2"):
            self.stats["l2_selected"] += 1
        if len(plan.actions) > 1 and any(a != plan.actions[0] for a in plan.actions[1:]):
            self.carry = plan
        self.prev_health = snap.health.copy()
        self.prev_demand = snap.demand
        ranking = [
            {"rank": k + 1, "plan_id": p.plan_id, "source": p.source, **r.metrics()}
            for k, (p, r) in enumerate(ctx.scored[: self.keep_ranking])
        ]
        notes = dict(ctx.notes)
        recheck = getattr(self.forecaster, "recheck_requests", None)
        if recheck:
            notes["recheck"] = sorted({r.get("recheck_room") for r in recheck if r.get("recheck_room")})
            recheck.clear()
        return Decision(
            action=plan.first(),
            plan=plan,
            predicted=res,
            ranking=ranking,
            n_candidates=ctx.counts,
            rejected=ctx.rejected,
            trigger=ctx.trigger,
            llm=ctx.llm,
            forecast={"demand": ctx.fc.demand.tolist(), "adjustments": ctx.fc.adjustments,
                      "conflicts": ctx.fc.conflicts},
            notes=notes,
        )

    def decide(self, snap: Snapshot) -> Decision:
        ctx = self.stage_forecast(snap)
        for stage in (self.stage_pool, self.stage_l2, self.stage_compile, self.stage_score, self.stage_tradeoff):
            ctx = stage(ctx)
        return self.stage_decision(ctx)


@dataclass
class PlanContext:
    snap: Snapshot
    fc: Forecast
    pool: list[CompiledPlan] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    hold_plan: CompiledPlan | None = None
    hold_r: RolloutResult | None = None
    trigger: str | None = None
    l2_raw: list[Any] = field(default_factory=list)
    scored: list[tuple[CompiledPlan, RolloutResult]] = field(default_factory=list)
    chosen_idx: int = 0
    llm: list[dict[str, Any]] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)


def make_policy(name: str, cfg: PlannerConfig, **kw: Any) -> Any:
    """Factory used by experiments: fixed | threshold | threshold_eff | planner | planner_nomove | arm_A..arm_D."""
    if name == "fixed":
        return FixedOpening(cfg)
    if name == "threshold":
        return DemandThreshold(cfg, "inventory")
    if name == "threshold_eff":
        return DemandThreshold(cfg, "efficiency")
    if name == "planner":
        return SimulationPlanner(cfg, forecaster=kw.get("forecaster"), name="planner")
    if name == "planner_nomove":
        return SimulationPlanner(replace(cfg, lambda_M=0.0), forecaster=kw.get("forecaster"), name="planner_nomove")
    if name == "arm_A":
        return SimulationPlanner(cfg, forecaster=kw.get("forecaster"), name="A")
    if name == "arm_B":
        return SimulationPlanner(cfg, forecaster=kw.get("forecaster"), use_rule=True, name="B")
    if name == "arm_C":
        return SimulationPlanner(cfg, forecaster=kw.get("forecaster"), l2=kw["l2"], name="C")
    if name == "arm_D":
        return SimulationPlanner(cfg, forecaster=kw.get("forecaster"), use_rule=True, l2=kw["l2"], name="D")
    raise KeyError(name)
