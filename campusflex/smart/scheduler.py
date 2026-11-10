"""Device schedulers: what every device of every room should do this tick (IoT dispatch).

* ``StaticScheduler`` – a conventional building: everything on at fixed settings while the
  space plan has the room open (AC 24 °C at full fan, full light, fresh air level 2,
  curtains half open, projector and sockets on).
* ``RuleScheduler`` – occupancy-based on/off with fixed comfort settings.
* ``SmartScheduler`` – targets from the learned profile of the room and slot (AI, when a
  ``ProfileStore`` is given) and sensor-based fine control (DSA, ``use_dsa``):
  pre-cooling lead from the identified thermal model, fan escalation when the AC cannot
  hold the setpoint, early AC stop before the room empties, daylight harvesting, glare
  control, CO₂ demand-controlled ventilation with hysteresis, and standby cut-off.

The space plan (which rooms are used, how many people, from the core planner) is known in
advance; occupancy within a tick comes from the people counter.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..config import COP, ROOM_BY_ID
from .devices import FAN_CAPACITY, DeviceState
from .dsa import Estimate, co2_after
from .physics import E_MAX_LUX, SUNSET_MIN

DEFAULT_PREFS = {"setpoint": 25.0, "fan": 2, "lux": 400.0, "curtain": 30, "projector": False}
CO2_TARGET = 900.0
VENT_DWELL_MIN = 6
OFF: dict[str, dict[str, Any]] = {"ac": {"setpoint": None, "fan": 0}, "light": {"level": 0},
                                  "vent": {"level": 0}, "projector": {"on": False}, "sockets": {"on": False}}


@dataclass
class RoomView:
    room: str
    minute: int
    slot: int
    open_plan: bool                    # the space plan uses this room now
    n_plan: int                        # planned occupants now
    n_next: int                        # planned occupants in the next ten minutes
    minutes_to_arrival: int | None     # when the room is empty: minutes until people are planned (≤ 30)
    state: DeviceState                 # applied device state (gateway view)
    t_out: float
    est: Estimate | None = None        # DSA estimate (filtered, anomaly-checked)
    raw: dict[str, Any] = field(default_factory=dict)   # latest raw telemetry
    ac_available: bool = True
    user_recent: dict[str, dict[str, Any]] = field(default_factory=dict)   # last manual setting per device, this slot


def _ceil10(pct: float) -> int:
    return int(min(100, max(0, math.ceil(pct / 10.0 - 1e-9) * 10)))


class StaticScheduler:
    name = "static"

    def decide(self, v: RoomView) -> dict[str, dict[str, Any]]:
        if not v.open_plan:
            return dict(OFF)
        return {"ac": {"setpoint": 24.0, "fan": 3}, "light": {"level": 100}, "vent": {"level": 2},
                "curtain": {"open_pct": 50}, "projector": {"on": True}, "sockets": {"on": True}}


class RuleScheduler:
    name = "rule"

    def decide(self, v: RoomView) -> dict[str, dict[str, Any]]:
        if v.n_plan <= 0:
            return dict(OFF)
        dens = v.n_plan / ROOM_BY_ID[v.room].seats
        return {"ac": {"setpoint": 25.0, "fan": 2}, "light": {"level": 100},
                "vent": {"level": 1 if dens < 0.3 else 2 if dens < 0.7 else 3},
                "curtain": {"open_pct": 50 if v.minute < SUNSET_MIN else 0},
                "projector": {"on": True}, "sockets": {"on": True}}


class SmartScheduler:
    def __init__(self, store: Any = None, use_dsa: bool = True, name: str | None = None):
        self.store = store
        self.use_dsa = use_dsa
        self.name = name or ("smart" if store is not None and use_dsa else
                             "smart_profiles_only" if store is not None else "smart_dsa_only")
        self._vent_changed: dict[str, int] = {}

    def prefs(self, v: RoomView) -> dict[str, Any]:
        """Profile (or default) settings, overridden by what people set by hand earlier in this
        slot – the scheduler does not undo a manual choice once its hold expires."""
        p = dict(DEFAULT_PREFS) if self.store is None else self.store.recommend(v.room, v.slot)
        u = v.user_recent
        if "ac" in u and u["ac"].get("setpoint") is not None:
            p["setpoint"], p["fan"] = float(u["ac"]["setpoint"]), int(u["ac"]["fan"])
        if "curtain" in u:
            p["curtain"] = int(u["curtain"]["open_pct"])
        if "projector" in u:
            p["projector"] = bool(u["projector"]["on"])
        if "light" in u and "lux_after" in u["light"]:
            p["lux"] = float(u["light"]["lux_after"])
        return p

    # ------------------------------------------------------------ helpers
    def _temp(self, v: RoomView) -> float | None:
        return v.est.temp_c if (self.use_dsa and v.est is not None) else v.raw.get("temp_c")

    def _co2(self, v: RoomView) -> float | None:
        return v.est.co2_ppm if (self.use_dsa and v.est is not None) else v.raw.get("co2_ppm")

    def _lead_min(self, v: RoomView, sp: float, fan: int) -> int:
        """Pre-cooling lead: minutes the AC needs to bring the room to ``sp`` (identified model)."""
        if not self.use_dsa or v.est is None:
            return 10
        T = v.est.temp_c
        spec = ROOM_BY_ID[v.room]
        C, R = spec.C, spec.R                       # design values until the room's model is identified
        ident = v.est.identified
        if ident and 0.3 < ident["C"] < 20.0 and 0.2 < ident["R"] < 10.0:
            C, R = ident["C"], ident["R"]
        load = (v.t_out - T) / R
        cap = COP * spec.P_ac * FAN_CAPACITY[max(fan, 1)] - load
        if T <= sp or cap <= 0:
            return 4
        return int(min(30, max(4, (T - sp) * C / cap * 60.0 + 4)))

    # ------------------------------------------------------------- decide
    def decide(self, v: RoomView) -> dict[str, dict[str, Any]]:
        p = self.prefs(v)
        sp, fan = float(p["setpoint"]), int(p["fan"])
        occupied = v.n_plan > 0
        soon = (not occupied and v.minutes_to_arrival is not None
                and v.minutes_to_arrival <= self._lead_min(v, sp, fan))
        out: dict[str, dict[str, Any]] = {}
        if not occupied and not soon:
            out = dict(OFF)
            co2 = self._co2(v)
            if self.use_dsa and co2 is not None and co2 > 650:
                out["vent"] = {"level": 1}                     # flush the room after use
            if v.minute < SUNSET_MIN and ROOM_BY_ID[v.room].building == "B":
                out["curtain"] = {"open_pct": 0}               # keep the low sun out of an empty room
            return out
        # ---- air conditioning
        if v.ac_available:
            T = self._temp(v)
            if self.use_dsa and occupied and v.n_next == 0 and (T is None or T <= sp + 0.5):
                out["ac"] = {"setpoint": None, "fan": 0}       # people leave within ten minutes: coast
            else:
                if self.use_dsa and T is not None and T - sp > 0.5 and v.state.ac_on():
                    # cannot hold the setpoint: more air – one level above a fan speed people are known
                    # to accept (set by hand this slot, or learned for this room), never above an unknown
                    known = ("ac" in v.user_recent and v.user_recent["ac"].get("setpoint") is not None) or \
                        (p.get("evidence", {}).get("fan", 0.0) >= 1.0)
                    cap = int(p["fan"]) + 1 if known else int(p["fan"])
                    fan = max(fan, min(3, cap, v.state.ac_fan + 1))
                out["ac"] = {"setpoint": sp, "fan": fan}
        # ---- projector, curtains, light
        projector = bool(p["projector"]) and occupied
        out["projector"] = {"on": projector}
        daylight = (v.est.daylight_lux if (self.use_dsa and v.est is not None)
                    else (300.0 if v.minute < SUNSET_MIN else 0.0))
        curtain = int(p["curtain"])
        if projector and daylight * curtain / 100.0 > 150.0:
            curtain = 0                                        # avoid glare on the screen
        out["curtain"] = {"open_pct": curtain}
        if occupied:
            target = float(p["lux"])
            from_daylight = daylight * curtain / 100.0 if self.use_dsa else 0.0
            out["light"] = {"level": _ceil10((target - from_daylight) / E_MAX_LUX * 100.0)}
        else:
            out["light"] = {"level": 0}
        out["sockets"] = {"on": True}
        # ---- fresh air
        out["vent"] = {"level": self._vent(v, occupied)}
        return out

    def _vent(self, v: RoomView, occupied: bool) -> int:
        cur = v.state.vent
        co2 = self._co2(v)
        if co2 is None:
            return cur
        floor = 1 if occupied else 0
        if self.use_dsa:
            n = max(v.n_plan, v.n_next)
            level = next((L for L in range(floor, 4) if co2_after(co2, n, L, v.room, 10) <= CO2_TARGET), 3)
            last = self._vent_changed.get(v.room, -10**6)
            if level < cur and v.minute - last < VENT_DWELL_MIN:
                level = cur                                    # hysteresis: do not step down too soon
        else:
            level = min(3, cur + 1) if co2 >= 1000 else max(floor, cur - 1) if co2 <= 700 else max(floor, cur)
        if level != cur:
            self._vent_changed[v.room] = v.minute
        return level
