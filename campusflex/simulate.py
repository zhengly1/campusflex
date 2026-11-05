"""``simulate_plan``: rollout of a compiled plan with the planner's (nominal) model, and the
lexicographic score of Equation 4:

    score(a) = ( U(a),  E(a) + λT V(a) + λM M(a) + λS K(a) + λΩ Ω(a) )

U  unserved person-minutes, E kWh, V occupied degree-person-hours (interval midpoints),
M relocations, K room-state switches, Ω occupied deviation over the window after the
rollout assuming the plan's final state is held.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import DT_H, N_STEPS, STEP_MIN, PlannerConfig
from .forecast import Forecast
from .model import (
    EFFICIENCY_ORDER,
    INVENTORY_ORDER,
    NOMINAL,
    ThermalParams,
    assign,
    deviation,
    power_kw,
    smart_fill_order,
    step_temps,
    thermostat_u,
)
from .plans import CompiledPlan, StepAction
from .state import Snapshot
from .validator import later_step_violations

U_TOL = 1e-6


def fill_order_for(action: StepAction, temps: np.ndarray, health: np.ndarray) -> list[int]:
    if action.fill == "inventory":
        return INVENTORY_ORDER
    if action.fill == "efficiency":
        return EFFICIENCY_ORDER
    return smart_fill_order(temps, health)


@dataclass(frozen=True, order=True)
class Score:
    U: float
    J: float

    def key(self) -> tuple[float, float]:
        return (round(self.U / U_TOL) * U_TOL, self.J)


@dataclass
class RolloutResult:
    plan_id: str
    source: str
    U: float = 0.0          # unserved person-minutes
    E: float = 0.0          # kWh
    V: float = 0.0          # degree-person-hours (occupied)
    M: int = 0              # relocations
    K: int = 0              # switches
    Omega: float = 0.0
    LS: int = 0             # loss-of-seat events
    occ_h: float = 0.0      # occupied person-hours within the horizon
    first_V: float = 0.0
    first_E: float = 0.0
    max_power_kw: float = 0.0
    power_cap_violated: bool = False
    budget_violated: bool = False     # L1 energy budget (pro-rata) exceeded
    later_violations: list[Any] = field(default_factory=list)   # later plan steps the validator would refuse
    score: Score | None = None
    temps: np.ndarray | None = None
    counts: np.ndarray | None = None

    @property
    def dev_mean(self) -> float:
        return self.V / self.occ_h if self.occ_h > 0 else 0.0

    def metrics(self) -> dict[str, Any]:
        return {
            "U": round(self.U, 4),
            "E": round(self.E, 4),
            "V": round(self.V, 4),
            "M": int(self.M),
            "K": int(self.K),
            "Omega": round(self.Omega, 4),
            "dev_mean": round(self.dev_mean, 4),
            "score_U": round(self.score.U, 4) if self.score else None,
            "score_J": round(self.score.J, 4) if self.score else None,
        }


def score_of(r: RolloutResult, cfg: PlannerConfig, include_omega: bool = True) -> Score:
    J = r.E + cfg.lambda_T * r.V + cfg.lambda_M * r.M + cfg.lambda_S * r.K
    if include_omega:
        J += cfg.lambda_Omega * r.Omega
    return Score(U=r.U, J=J)


def rollout(
    snap: Snapshot,
    plan: CompiledPlan,
    fc: Forecast,
    cfg: PlannerConfig,
    params: ThermalParams = NOMINAL,
    keep_traj: bool = False,
) -> RolloutResult:
    # The evening ends at 22:00: no energy, deviation or Ω is counted after the last interval.
    left = max(1, N_STEPS - snap.step)
    H = min(cfg.horizon_steps, left)
    W = min(cfg.omega_steps, left - H)
    T = snap.temps.astype(float).copy()
    n = snap.counts.astype(int).copy()
    open_prev = snap.open.astype(bool).copy()
    health = snap.health
    cap = snap.capacity
    res = RolloutResult(plan_id=plan.plan_id, source=plan.source)
    temps_tr = [T.copy()] if keep_traj else None
    counts_tr = [] if keep_traj else None
    hold = plan.action_at(H)  # final state, no moves
    for k in range(H + W):
        a = plan.action_at(k) if k < H else hold
        d = int(fc.demand[min(k, len(fc.demand) - 1)])
        tout = float(fc.t_out[min(k, len(fc.t_out) - 1)])
        mask = a.open_mask()
        if 0 < k < min(H, len(plan.actions)):
            res.later_violations += later_step_violations(a, open_prev, n, cap, T, health,
                                                          snap.clock_min + k * STEP_MIN, cfg, f"+{k * STEP_MIN}")
        ar = assign(n, open_prev, mask, cap, d, T, fill_order_for(a, T, health), a.move_indices())
        n = ar.counts
        u = thermostat_u(T, tout, n, a.setpoint_array(), health, NOMINAL)
        T2 = step_temps(T, tout, n, u, params)
        v_int = float(np.sum(n * deviation(0.5 * (T + T2), cfg.band_low_c, cfg.band_high_c)) * DT_H)
        if k < H:
            p = power_kw(u, mask.astype(float), params)
            e_int = float(p.sum() * DT_H)
            res.U += ar.unserved * STEP_MIN
            res.E += e_int
            res.V += v_int
            res.M += ar.moved
            res.K += ar.switches
            res.LS += ar.loss_of_seat
            res.occ_h += float(n.sum()) * DT_H
            res.max_power_kw = max(res.max_power_kw, float(p.sum()))
            if k == 0:
                res.first_V, res.first_E = v_int, e_int
        else:
            res.Omega += v_int
        if keep_traj:
            temps_tr.append(T2.copy())
            counts_tr.append(n.copy())
        T, open_prev = T2, mask
    if cfg.power_cap_kw is not None and res.max_power_kw > cfg.power_cap_kw + 1e-9:
        res.power_cap_violated = True
    if cfg.energy_budget_kwh is not None:
        # Pro-rata evening budget up to the end of the horizon (energy already used included).
        allowed = cfg.energy_budget_kwh * min(N_STEPS, snap.step + H) / N_STEPS
        res.budget_violated = getattr(snap, "energy_used_kwh", 0.0) + res.E > allowed + 1e-9
    res.score = score_of(res, cfg)
    if keep_traj:
        res.temps = np.array(temps_tr)
        res.counts = np.array(counts_tr)
    return res
