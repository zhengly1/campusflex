"""Campus environment implementing the tool contract (Section 3.5):

    get_state      → versioned snapshot (temperatures read from device telemetry)
    simulate_plan  → rollout with the planner's nominal model
    validate_plan  → violations
    apply_plan     → per-room outcomes; idempotency keys; ledger changes only after ack
    advance_time   → next observation in ten minutes

The environment's physics may use perturbed parameters (model mismatch), so predictions
and outcomes can differ – which is what the V-gap log measures.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import (
    BAND_HIGH_C,
    BAND_LOW_C,
    DT_H,
    N_ROOMS,
    N_STEPS,
    ROOM_IDS,
    STEP_MIN,
    PlannerConfig,
    clock_str,
    step_to_clock,
)
from .devices import Command, DeviceLayer, DirectDeviceLayer, Plant
from .forecast import Forecast
from .ledger import Ledger
from .model import NOMINAL, ThermalParams, assign, deviation, power_kw, step_temps, thermostat_u
from .plans import CompiledPlan, StepAction
from .scenarios import Scenario, _rng
from .simulate import RolloutResult, fill_order_for, rollout
from .state import Snapshot
from .validator import Violation, validate_action, validate_plan

STALE_AFTER_MIN = 2 * STEP_MIN      # a reading older than two intervals is stale


@dataclass
class ApplyResult:
    executed: bool
    duplicate: bool = False
    rejected: list[dict[str, str]] = field(default_factory=list)
    outcomes: dict[str, str] = field(default_factory=dict)       # room → "ack" | "no_ack" | "ac_offline"
    effective: dict[str, Any] | None = None
    idempotency_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "executed": self.executed,
            "duplicate": self.duplicate,
            "rejected": self.rejected,
            "outcomes": self.outcomes,
            "effective": self.effective,
            "idempotency_key": self.idempotency_key,
        }


class CampusEnv:
    def __init__(
        self,
        scenario: Scenario,
        cfg: PlannerConfig,
        device_layer: DeviceLayer | None = None,
    ):
        self.sc = scenario
        self.cfg = cfg
        self.true_params: ThermalParams = ThermalParams.perturbed(scenario.mismatch)
        self.plant = Plant.initial()
        self.capacity = scenario.capacities()
        self.step = 0
        self.version = 1
        self.ledger = Ledger(_rng(scenario.name, scenario.seed, "departures"), cfg.departure_mode)
        self.believed_open = np.zeros(N_ROOMS, dtype=bool)        # updated only after ack
        self.believed_setpoint = np.full(N_ROOMS, np.nan)
        self.steps_since_switch = np.full(N_ROOMS, 10**6, dtype=int)
        self.device = device_layer or DirectDeviceLayer()
        self.device.bind(self.plant)
        self.done = False
        self._key_results: dict[str, ApplyResult] = {}
        self._pending: dict[str, Any] | None = None      # assignment for the current interval
        self.step_logs: list[dict[str, Any]] = []
        self.audit: list[dict[str, Any]] = []            # every apply_plan call
        self.counters = {"unvalidated_executed": 0, "rejected": 0, "duplicates": 0, "stale": 0, "no_ack": 0,
                         "drift": 0, "deferred_close": 0, "dropped_moves": 0}
        self.rechecks: list[str] = []
        # Telemetry sequence number per room at the time of its last command: only readings
        # taken after a command may correct the believed state (they reflect its effect).
        self._seq_at_cmd: dict[str, int] = {}
        # Room → telemetry seq at which the device refused cooling (``ac_offline`` ack).
        self._offline_ack_seq: dict[str, int] = {}
        self.drift_log: list[dict[str, Any]] = []
        self.ledger.set_demand(int(scenario.demand[0]))
        self.plant.health[:] = scenario.health[0]
        self.device.publish(self.clock)

    # ----------------------------------------------------------------- helpers
    @property
    def clock(self) -> int:
        return step_to_clock(self.step)

    def request_recheck(self, room: str) -> None:
        """L3 occupant reports never move students; they trigger a sensor re-read."""
        if room in ROOM_IDS and room not in self.rechecks:
            self.rechecks.append(room)

    # ------------------------------------------------------------- get_state
    def get_state(self) -> Snapshot:
        """Versioned snapshot built only from device telemetry and acknowledged commands.

        * A room whose reading is older than two intervals is flagged ``stale``; a room that
          never reported has unknown temperature (outdoor temperature is assumed) and health 0.
        * After an ``ac_offline`` ack the room's AC is believed offline until a newer reading
          says otherwise.
        * A reading taken after the last command that disagrees with the believed open/closed
          state (e.g. the command was applied but its ack was lost) corrects the belief and is
          counted as ``drift``.
        """
        if self.rechecks:
            self.device.publish(self.clock)
        tel = self.device.latest()
        temps = np.empty(N_ROOMS)
        health = np.empty(N_ROOMS)
        stale = np.zeros(N_ROOMS, dtype=bool)
        for i, r in enumerate(ROOM_IDS):
            t = tel.get(r)
            if t is None:
                temps[i], health[i], stale[i] = float(self.sc.t_out[self.step]), 0.0, True
                continue
            temps[i], health[i] = t.temp_c, t.health
            stale[i] = (self.clock - t.ts_min) > STALE_AFTER_MIN
            off = self._offline_ack_seq.get(r)
            if off is not None:
                if t.seq > off and t.health > 0 and not stale[i]:
                    del self._offline_ack_seq[r]          # newer reading: unit is back
                else:
                    health[i] = 0.0
            if t.seq > self._seq_at_cmd.get(r, -1) and not stale[i]:
                self._reconcile(i, t)
        snap = Snapshot(
            version=self.version,
            step=self.step,
            clock_min=self.clock,
            demand=int(self.sc.demand[self.step]),
            t_out=float(self.sc.t_out[self.step]),
            temps=temps,
            health=health,
            open=self.believed_open.copy(),
            counts=self.ledger.counts(),
            capacity=self.capacity.copy(),
            steps_since_switch=self.steps_since_switch.copy(),
            movable=self.ledger.movable(self.cfg, self.step, self.clock),
            notices=self.sc.released_notices(self.clock),
            stale=stale,
            rechecks=list(self.rechecks),
            energy_used_kwh=float(sum(l["E"] for l in self.step_logs)),
        )
        self.rechecks.clear()
        return snap

    def _reconcile(self, i: int, t: Any) -> None:
        r = ROOM_IDS[i]
        sp_t = getattr(t, "setpoint", None)
        sp_t = np.nan if sp_t is None else float(sp_t)
        open_drift = bool(t.open) != bool(self.believed_open[i])
        sp_drift = bool(t.open) and not (np.isnan(sp_t) and np.isnan(self.believed_setpoint[i])) \
            and not (sp_t == self.believed_setpoint[i])
        if not (open_drift or sp_drift):
            return
        self.counters["drift"] += 1
        self.drift_log.append({"step": self.step, "room": r, "believed_open": bool(self.believed_open[i]),
                               "reported_open": bool(t.open), "occupants_in_ledger": int(self.ledger.counts()[i])})
        self.believed_open[i] = bool(t.open)
        self.believed_setpoint[i] = sp_t if t.open else np.nan
        self.version += 1     # belief changed: plans built on the previous snapshot are stale

    # ---------------------------------------------------------- simulate_plan
    def simulate_plan(self, plan: CompiledPlan, fc: Forecast, snap: Snapshot | None = None) -> RolloutResult:
        return rollout(snap or self.get_state(), plan, fc, self.cfg, NOMINAL)

    # ---------------------------------------------------------- validate_plan
    def validate_plan(self, plan: CompiledPlan, snap: Snapshot | None = None) -> list[Violation]:
        return validate_plan(plan, snap or self.get_state(), self.cfg)

    # ------------------------------------------------------------- apply_plan
    def apply_plan(
        self,
        action: StepAction,
        state_version: int,
        idempotency_key: str,
        meta: dict[str, Any] | None = None,
    ) -> ApplyResult:
        """Commit one action. Only the deterministic planner node may call this."""
        entry = {"step": self.step, "key": idempotency_key, "action": action.to_dict(), "meta": meta or {}}
        if idempotency_key in self._key_results:
            self.counters["duplicates"] += 1
            prev = self._key_results[idempotency_key]
            res = ApplyResult(executed=False, duplicate=True, outcomes=prev.outcomes,
                              effective=prev.effective, idempotency_key=idempotency_key)
            self.audit.append({**entry, "result": "duplicate"})
            return res
        snap = self.get_state()            # may correct the belief from newer telemetry (bumps version)
        if state_version != self.version:
            self.counters["stale"] += 1
            res = ApplyResult(False, rejected=[{"code": "stale_snapshot",
                                                "detail": f"plan for v{state_version}, state is v{self.version}"}],
                              idempotency_key=idempotency_key)
            self.audit.append({**entry, "result": "stale"})
            return res
        if self._pending is not None:
            res = ApplyResult(False, rejected=[{"code": "already_committed", "detail": "one commit per step"}],
                              idempotency_key=idempotency_key)
            self.audit.append({**entry, "result": "already_committed"})
            return res
        viol = validate_action(action, snap, self.cfg)
        if viol:
            self.counters["rejected"] += 1
            res = ApplyResult(False, rejected=[v.to_dict() for v in viol], idempotency_key=idempotency_key)
            self.audit.append({**entry, "result": "rejected", "violations": res.rejected})
            return res

        # Send per-room commands; the believed state changes only for acknowledged rooms.
        # Make-before-break: rooms that stay open or open first; rooms that close only after
        # every newly opened room has acknowledged, so a lost ack never strands occupants.
        new_open = action.open_mask()
        new_sp = action.setpoint_array()
        tel = self.device.latest()

        def cmd(i: int) -> Command:
            r = ROOM_IDS[i]
            sp = None if np.isnan(new_sp[i]) else float(new_sp[i])
            self._seq_at_cmd[r] = tel[r].seq if r in tel else -1
            return Command(f"{idempotency_key}:{r}", r, bool(new_open[i]), sp, self.clock)

        closing = [i for i in range(N_ROOMS) if self.believed_open[i] and not new_open[i]]
        opening = [i for i in range(N_ROOMS) if new_open[i] and not self.believed_open[i]]
        acks = dict(self.device.command([cmd(i) for i in range(N_ROOMS) if i not in closing], self.clock))
        outcomes: dict[str, str] = {}
        if closing:
            if all(acks.get(ROOM_IDS[i]) is not None and acks[ROOM_IDS[i]].open for i in opening):
                acks.update(self.device.command([cmd(i) for i in closing], self.clock))
            else:
                for i in closing:
                    outcomes[ROOM_IDS[i]] = "deferred"
                    self.counters["deferred_close"] += 1
        eff_open = self.believed_open.copy()
        eff_sp = self.believed_setpoint.copy()
        for i, r in enumerate(ROOM_IDS):
            if r in outcomes:
                continue
            ack = acks.get(r)
            if ack is None:
                outcomes[r] = "no_ack"
                self.counters["no_ack"] += 1
                continue
            outcomes[r] = "ack" if ack.ok else ack.reason
            if ack.reason == "out_of_order":
                continue
            if ack.reason == "ac_offline":
                self._offline_ack_seq[r] = tel[r].seq if r in tel else -1
            eff_open[i] = ack.open
            eff_sp[i] = np.nan if ack.setpoint is None else ack.setpoint
        switched = eff_open != self.believed_open
        # Moves are executed only between rooms that are effectively open after the acks.
        moves = tuple(m for m in action.moves if eff_open[ROOM_IDS.index(m[1])])
        self.counters["dropped_moves"] += len(action.moves) - len(moves)
        eff_action = StepAction(
            open=frozenset(r for r, o in zip(ROOM_IDS, eff_open) if o),
            setpoints=tuple((r, float(eff_sp[i])) for i, r in enumerate(ROOM_IDS) if not np.isnan(eff_sp[i])),
            moves=moves,
            fill=action.fill,
        )
        self._assign(eff_action, eff_open, switched)
        self.believed_open, self.believed_setpoint = eff_open, eff_sp
        res = ApplyResult(True, outcomes=outcomes, effective=eff_action.to_dict(), idempotency_key=idempotency_key)
        self._key_results[idempotency_key] = res
        self.audit.append({**entry, "result": "executed", "outcomes": outcomes})
        return res

    def _assign(self, a: StepAction, eff_open: np.ndarray, switched: np.ndarray) -> None:
        counts = self.ledger.counts()
        temps = self.plant.T
        ar = assign(
            counts,
            self.believed_open,
            eff_open,
            self.capacity,
            self.ledger.present(),
            temps,
            fill_order_for(a, temps, self.plant.health),
            a.move_indices(),
            depart_proportional=False,
        )
        self.ledger.apply(ar, self.step)
        self.plant.occupancy[:] = ar.counts
        self.steps_since_switch = np.where(switched, 0, self.steps_since_switch)
        self._pending = {"assign": ar, "action": a}

    # ----------------------------------------------------------- advance_time
    def advance_time(self, log_extra: dict[str, Any] | None = None) -> Snapshot | None:
        if self.done:
            return None
        if self._pending is None:      # nothing committed: devices hold their last state
            hold = StepAction(open=frozenset(r for r, o in zip(ROOM_IDS, self.believed_open) if o),
                              setpoints=tuple((r, float(v)) for r, v in zip(ROOM_IDS, self.believed_setpoint)
                                              if not np.isnan(v)))
            self._assign(hold, self.believed_open.copy(), np.zeros(N_ROOMS, dtype=bool))
            self._pending["held"] = True
        ar = self._pending["assign"]
        t = self.step
        tout = float(self.sc.t_out[t])
        n = ar.counts
        T0 = self.plant.T.copy()
        u = thermostat_u(T0, tout, n, self.plant.setpoint, self.plant.health, NOMINAL)
        T1 = step_temps(T0, tout, n, u, self.true_params)
        lights = self.plant.open.astype(float)
        p = power_kw(u, lights, self.true_params)
        # Realized metrics always use the declared 24–27 °C band (Eq. 5 yardstick), even when
        # L1 compiled a different band for the planner's objective.
        dev = deviation(0.5 * (T0 + T1), BAND_LOW_C, BAND_HIGH_C)
        d = int(self.sc.demand[t])
        log = {
            "step": t,
            "clock": clock_str(self.clock),
            "state_version": self.version,
            "demand": d,
            "served": int(n.sum()),
            "unserved": max(0, d - int(n.sum())),
            "counts": n.tolist(),
            "open": self.plant.open.astype(int).tolist(),
            "u": np.round(u, 4).tolist(),
            "T_start": np.round(T0, 4).tolist(),
            "T_end": np.round(T1, 4).tolist(),
            "t_out": tout,
            "health": self.plant.health.tolist(),
            "E": float(p.sum() * DT_H),
            "V": float(np.sum(n * dev) * DT_H),
            "occ_h": float(n.sum() * DT_H),
            "M": int(ar.moved),
            "forced_moves": int(ar.forced),
            "K": int(ar.switches),
            "LS": int(ar.loss_of_seat),
            "held": bool(self._pending.get("held", False)),
            "violations": self._hard_violations(n, u),
            "drift": int(self.counters["drift"]),
        }
        if log_extra:
            log.update(log_extra)
        self.step_logs.append(log)
        # advance
        self.plant.T = T1
        self.steps_since_switch = self.steps_since_switch + 1
        self._pending = None
        self.step += 1
        self.version += 1
        if self.step >= N_STEPS:
            self.done = True
            return None
        self.plant.health[:] = self.sc.health[self.step]
        self.ledger.set_demand(int(self.sc.demand[self.step]))
        self.device.publish(self.clock)
        return self.get_state()

    def _hard_violations(self, n: np.ndarray, u: np.ndarray) -> int:
        """Eq. (1) audit: capacity, occupancy of closed rooms, cooling from offline units."""
        bad = 0
        bad += int(np.sum(n > self.capacity * self.plant.open))        # physical state, not belief
        bad += int(np.sum(u > self.plant.health + 1e-12))
        bad += int(n.sum() > int(self.sc.demand[self.step]))
        return bad

    # ------------------------------------------------------------ utilities
    def save_state(self) -> "CampusEnv":
        """Deep copy for counterfactual replays (device layer must be in-process)."""
        return copy.deepcopy(self)
