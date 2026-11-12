"""One evening of the smart device layer, and many evenings for learning.

Every ``TICK_MIN`` minutes: the plant advances under the applied device states → the
sensors report over the bus → the DSA estimators update → occupants adjust what bothers
them (wall-panel commands over the bus; the profile learner sees each adjustment and the
scheduler leaves that device alone for ``HOLD_MIN``) → the scheduler's desired settings are
sent as commands for the devices that differ from the applied state.

The space plan (rooms, occupants per room, outdoor temperature, AC health) comes from a
core episode (``day_plan``), so the core planner keeps deciding where people sit and the
smart layer decides how each device runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import END_MIN, ROOM_BY_ID, ROOM_IDS, START_MIN, STEP_MIN, PlannerConfig, clock_str
from .devices import DEVICES, DeviceState
from .dsa import RoomEstimator, score_detection
from .iot import SmartIoT
from .occupants import Occupants, campus_habits
from .physics import DT_H, E_MAX_LUX, TICK_MIN, SmartPlant
from .profiles import ProfileStore, slot_label, slot_of
from .scheduler import RoomView
from .sensors import SensorFault, SensorSuite

HOLD_MIN = 30
SAFETY_CO2_PPM = 1500.0
DEADBAND = {"setpoint": 0.25}


@dataclass
class DayPlan:
    """Space plan of one evening from the core (10-minute steps)."""

    day: int
    scenario: str
    seed: int
    counts: list[list[int]]          # [step][room]
    open: list[list[int]]            # [step][room]
    t_out: list[float]               # N_STEPS + 1 values
    health: list[list[float]]        # [step][room]
    faults: list[SensorFault] = field(default_factory=list)


def day_plan(day: int, scenario: str, seed: int, policy: str = "arm_A",
             faults: list[SensorFault] | None = None) -> DayPlan:
    """Run the core planner for one evening and keep its space plan."""
    from ..episode import run_episode
    from ..policies import make_policy
    from ..scenarios import make_scenario

    cfg = PlannerConfig()
    sc = make_scenario(scenario, seed)
    r = run_episode(sc, make_policy(policy, cfg), cfg)
    return DayPlan(day, scenario, seed, [list(map(int, s["counts"])) for s in r.steps],
                   [list(map(int, s["open"])) for s in r.steps], [float(x) for x in sc.t_out],
                   [list(map(float, h)) for h in sc.health], list(faults or []))


def _t_out(plan: DayPlan, minute: int) -> float:
    x = (minute - START_MIN) / STEP_MIN
    i = min(int(x), len(plan.t_out) - 2)
    f = x - i
    return plan.t_out[i] * (1 - f) + plan.t_out[i + 1] * f


def _differs(state: DeviceState, device: str, value: dict[str, Any]) -> bool:
    cur = state.value(device)
    if device == "ac":
        a, b = cur["setpoint"], value["setpoint"]
        if (a is None) != (b is None):
            return True
        if a is not None and abs(float(a) - float(b)) > DEADBAND["setpoint"]:
            return True
        return cur["fan"] != value["fan"] and b is not None
    return cur != value


def run_day(plan: DayPlan, scheduler: Any, store: ProfileStore | None = None, learn: bool = False,
            habits: dict | None = None, seed: int | None = None, bus: Any = None,
            models: dict[str, Any] | None = None, trace: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Simulate one evening; returns metrics, adjustments, anomaly flags and the identified models.

    ``models`` (room → ``ThermalRLS``) carries the identified thermal models from evening to
    evening; without it every evening starts identification from scratch. A ``trace`` list
    receives one record per tick (``trace_tick``); recording only reads the state."""
    from ..iot.bus import InProcessBus

    seed = plan.seed if seed is None else seed
    habits = habits or campus_habits()
    plant = SmartPlant()
    iot = SmartIoT(bus or InProcessBus(), plant, owns_bus=bus is None)
    sensors = SensorSuite(seed * 7 + 1, plan.faults)
    est = {r: RoomEstimator(r, (models or {}).get(r)) for r in ROOM_IDS}
    if models is not None:
        for r in ROOM_IDS:
            models.setdefault(r, est[r].rls)
    occ = Occupants(habits, plan.day, seed * 13 + 5)
    holds: dict[tuple[str, str], int] = {}
    ac_available = {r: True for r in ROOM_IDS}
    empty_since: dict[str, int] = {}
    energy = {d: 0.0 for d in DEVICES}
    disc = {"thermal": 0.0, "light": 0.0, "co2": 0.0, "glare": 0.0}
    person_min = 0.0
    adjustments: list[dict[str, Any]] = []
    slot_acc: dict[tuple[str, int], dict[str, Any]] = {}
    user_recent: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    sched_cmds = 0
    try:
        for minute in range(START_MIN, END_MIN, TICK_MIN):
            k = (minute - START_MIN) // STEP_MIN
            slot = slot_of(minute)
            if minute > START_MIN and slot != slot_of(minute - TICK_MIN):
                holds.clear()          # a new session: manual settings of the last one no longer bind
            t_out = _t_out(plan, minute)
            occupancy = {r: plan.counts[k][i] for i, r in enumerate(ROOM_IDS)}
            for i, r in enumerate(ROOM_IDS):
                plant.health[r] = plan.health[k][i]
            recs = plant.step(minute, t_out, occupancy)
            iot.publish_sensors([sensors.read(recs[r]) for r in ROOM_IDS])
            ests = {r: est[r].update(iot.telemetry[r], t_out, iot.applied[r]) for r in ROOM_IDS}
            during = {r: plant.devices[r].to_dict() for r in ROOM_IDS} if trace is not None else {}
            events: list[dict[str, Any]] = []
            # ------------------------------------------------------------ metrics
            for r in ROOM_IDS:
                rec, n = recs[r], occupancy[r]
                for d, kw in rec.power_kw.items():
                    energy[d] += kw * DT_H
                if n > 0:
                    person_min += n * TICK_MIN
                    for key, bad in occ.discomfort(rec, plant.devices[r], slot).items():
                        if bad:
                            disc[key] += n * TICK_MIN
                    acc = slot_acc.setdefault((r, slot), {"ticks": 0, "adjusted": set(), "sp": [], "fan": [],
                                                          "lux": [], "curtain": [], "projector": []})
                    acc["ticks"] += 1
                    s = plant.devices[r]
                    if s.ac_on():
                        acc["sp"].append(s.ac_setpoint)
                        acc["fan"].append(s.ac_fan)
                    acc["lux"].append(rec.lux)
                    acc["curtain"].append(s.curtain)
                    acc["projector"].append(1.0 if s.projector else 0.0)
                    empty_since.pop(r, None)
                else:
                    empty_since.setdefault(r, minute)
            # ------------------------------------------------------- occupants act
            for r in ROOM_IDS:
                for a in occ.react(recs[r], plant.devices[r], slot, ac_available=ac_available[r]):
                    ack = iot.command(r, a.device, a.value, minute, source="user")
                    events.append(_event(r, a.device, a.value, "user", ack))
                    if ack is None or not ack.ok:
                        if ack is not None and ack.reason == "ac_offline":
                            ac_available[r] = False
                        continue
                    holds[(r, a.device)] = minute + HOLD_MIN
                    adjustments.append(a.to_dict())
                    acc = slot_acc.get((r, slot))
                    if acc is not None:
                        acc["adjusted"].add(a.device)
                    lux_after = None
                    if a.device == "light":
                        lux_after = (ests[r].daylight_lux * plant.devices[r].curtain / 100.0
                                     + a.value["level"] / 100.0 * E_MAX_LUX)
                    user_recent.setdefault((r, slot), {})[a.device] = (
                        {**a.value, "lux_after": lux_after} if lux_after is not None else dict(a.value))
                    if learn and store is not None:
                        store.observe_adjustment(r, slot, a.device, a.value, lux_after)
            # --------------------------------------------------------- scheduler
            for i, r in enumerate(ROOM_IDS):
                if r in empty_since and minute - empty_since[r] >= 10:
                    for d in DEVICES:
                        holds.pop((r, d), None)                  # nobody left to keep a manual setting for
                arrival = None if occupancy[r] > 0 else next(
                    ((kk - k) * STEP_MIN - (minute - START_MIN) % STEP_MIN
                     for kk in range(k + 1, min(len(plan.counts), k + 4)) if plan.counts[kk][i] > 0), None)
                view = RoomView(r, minute, slot_of(minute + (arrival or 0)), bool(plan.open[k][i]),
                                occupancy[r], plan.counts[min(k + 1, len(plan.counts) - 1)][i], arrival,
                                iot.applied[r], t_out, ests[r], iot.telemetry.get(r, {}), ac_available[r],
                                user_recent.get((r, slot_of(minute)), {}))
                for d, value in scheduler.decide(view).items():
                    held = holds.get((r, d), -1) > minute
                    safety = d == "vent" and ests[r].co2_ppm > SAFETY_CO2_PPM
                    if (held and not safety) or not _differs(iot.applied[r], d, value):
                        continue
                    if d == "ac" and value.get("setpoint") is not None and not ac_available[r]:
                        continue
                    ack = iot.command(r, d, value, minute)
                    events.append(_event(r, d, value, "scheduler", ack))
                    sched_cmds += 1
                    if ack is not None and not ack.ok and ack.reason == "ac_offline":
                        ac_available[r] = False
            if trace is not None:
                trace.append(trace_tick(minute, t_out, recs, iot.telemetry, ests, during, events, sensors,
                                        [f for e in est.values() for f in e.flags if f["minute"] == minute]))
        # ------------------------------------------------------ learn from the day
        if learn and store is not None:
            for (r, sl), acc in slot_acc.items():
                if acc["ticks"] * TICK_MIN < 20:
                    continue
                keep = {}
                if "ac" not in acc["adjusted"] and acc["sp"]:
                    keep["setpoint"], keep["fan"] = float(np.mean(acc["sp"])), float(np.mean(acc["fan"]))
                if "light" not in acc["adjusted"]:
                    keep["lux"] = float(np.mean(acc["lux"]))
                if "curtain" not in acc["adjusted"]:
                    keep["curtain"] = float(np.mean(acc["curtain"]))
                if "projector" not in acc["adjusted"]:
                    keep["projector"] = float(np.mean(acc["projector"]))
                store.observe_accepted(r, sl, keep)
            store.end_day()
    finally:
        iot.close()
    flags = [f for e in est.values() for f in e.flags]
    ident = {r: est[r].rls.identified() for r in ROOM_IDS}
    occupied_hours = sum(acc["ticks"] for acc in slot_acc.values()) * TICK_MIN / 60.0
    by_device: dict[str, int] = {}
    for a in adjustments:
        by_device[a["device"]] = by_device.get(a["device"], 0) + 1
    return {
        "day": plan.day, "scenario": plan.scenario, "seed": plan.seed, "scheduler": scheduler.name,
        "kwh": round(sum(energy.values()), 4), "kwh_by_device": {d: round(v, 4) for d, v in energy.items()},
        "person_min": person_min,
        "discomfort_pct": {k: round(100.0 * v / person_min, 3) if person_min else 0.0 for k, v in disc.items()},
        "adjustments": len(adjustments), "adjustments_by_device": by_device,
        "adjustments_per_room_hour": round(len(adjustments) / occupied_hours, 4) if occupied_hours else 0.0,
        "scheduler_commands": sched_cmds, "iot": dict(iot.stats),
        "faults": [f.to_dict() for f in plan.faults], "flags": flags,
        "detection": score_detection(flags, plan.faults),
        "identified": {r: ({k: round(v, 3) for k, v in x.items()} if x else None) for r, x in ident.items()},
        "adjustment_log": adjustments,
    }


def _event(room: str, device: str, value: dict[str, Any], by: str, ack: Any) -> dict[str, Any]:
    return {"room": room, "device": device, "value": dict(value), "by": by,
            "ok": bool(ack is not None and ack.ok), "reason": "" if ack is None else ack.reason}


def _r(x: float | None, nd: int) -> float | None:
    return None if x is None else round(float(x), nd)


def trace_tick(minute: int, t_out: float, recs: dict[str, Any], telemetry: dict[str, dict[str, Any]],
               ests: dict[str, Any], during: dict[str, dict[str, Any]], events: list[dict[str, Any]],
               sensors: SensorSuite, flags: list[dict[str, Any]]) -> dict[str, Any]:
    """One tick of an evening, per-room arrays in ``ROOM_IDS`` order: truth, sensor reading
    (None = dropout), DSA estimate, the device states applied during the tick, the commands
    sent at its end (``events``: by user or scheduler) and the faults flagged."""
    def col(f: Any) -> list[Any]:
        return [f(r) for r in ROOM_IDS]

    def tel(r: str, k: str, nd: int) -> float | None:
        return _r(telemetry.get(r, {}).get(k), nd)

    return {
        "minute": minute, "clock": clock_str(minute), "slot": slot_label(slot_of(minute)),
        "t_out": round(t_out, 2), "occupancy": col(lambda r: recs[r].occupancy),
        "temp_true": col(lambda r: _r(recs[r].temp_c, 2)), "temp_sensor": col(lambda r: tel(r, "temp_c", 2)),
        "temp_est": col(lambda r: _r(ests[r].temp_c, 2)),
        "co2_true": col(lambda r: _r(recs[r].co2_ppm, 0)), "co2_sensor": col(lambda r: tel(r, "co2_ppm", 0)),
        "co2_est": col(lambda r: _r(ests[r].co2_ppm, 0)),
        "lux_true": col(lambda r: _r(recs[r].lux, 0)), "lux_sensor": col(lambda r: tel(r, "lux", 0)),
        "lux_est": col(lambda r: _r(ests[r].lux, 0)),
        "setpoint": col(lambda r: during[r]["ac_setpoint"] if during[r]["ac_fan"] > 0 else None),
        "fan": col(lambda r: during[r]["ac_fan"] if during[r]["ac_setpoint"] is not None else 0),
        "light": col(lambda r: during[r]["light"]), "vent": col(lambda r: during[r]["vent"]),
        "curtain": col(lambda r: during[r]["curtain"]), "projector": col(lambda r: int(during[r]["projector"])),
        "sockets": col(lambda r: int(during[r]["sockets"])),
        "kw": col(lambda r: _r(sum(recs[r].power_kw.values()), 3)),
        "suspect": col(lambda r: dict(ests[r].faulty)),
        "fault_active": col(lambda r: {s: f.kind for s in ("temp", "co2", "lux")
                                       if (f := sensors.fault_at(r, s, minute)) is not None}),
        "events": events, "flags": [dict(f) for f in flags],
    }


def identification_error(days: list[dict[str, Any]]) -> dict[str, Any]:
    """Median relative error of the identified R and C against the room specifications."""
    err = {"R": [], "C": []}
    for d in days:
        for r, x in d["identified"].items():
            if x:
                err["R"].append(abs(x["R"] / ROOM_BY_ID[r].R - 1.0))
                err["C"].append(abs(x["C"] / ROOM_BY_ID[r].C - 1.0))
    return {k: {"n": len(v), "median_rel_error": round(float(np.median(v)), 4) if v else None} for k, v in err.items()}
