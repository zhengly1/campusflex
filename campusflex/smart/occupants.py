"""Simulated occupants: latent habits per room and time slot, manual adjustments, comfort.

The habits are the ground truth the profile learner has to discover. Each slot has a
campus-wide character (arrival with daylight, lecture with the projector, discussion,
late self-study); each room adds its own offsets (seeded, fixed for the campus); each day
adds noise. Occupants judge what they experience, not what the panel shows:

* temperature as felt, ``T − 0.4·(fan − 1)`` (air movement cools), against their preferred
  temperature ± 0.8 °C → they set their usual setpoint (too warm: never less fan than now;
  too cold: never more); a fan two levels away from their habit is reset to it;
* desk illuminance against their preferred lux ± 110 → they dim or brighten;
* the curtain more than 40 % away from their habit, or glare on the projector screen;
* the projector on/off against whether this slot uses it;
* CO₂ above 1200 ppm ("stuffy") → they turn the fresh air up.

A person adjusts a device at most once per 20 minutes and, when uncomfortable, acts with
probability 0.5 per two-minute tick.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import ROOM_IDS
from .devices import DeviceState
from .physics import E_MAX_LUX, TickRecord

# campus-wide character of each hourly slot
SLOT_BASE: tuple[dict[str, float], ...] = (
    {"setpoint": 25.5, "fan": 2, "lux": 420.0, "curtain": 60.0, "projector": 0.10},   # 18-19 arrival, daylight
    {"setpoint": 24.5, "fan": 2, "lux": 300.0, "curtain": 0.0, "projector": 0.90},    # 19-20 lecture
    {"setpoint": 25.0, "fan": 1, "lux": 380.0, "curtain": 0.0, "projector": 0.60},    # 20-21 discussion
    {"setpoint": 25.5, "fan": 1, "lux": 500.0, "curtain": 0.0, "projector": 0.10},    # 21-22 late self-study
)
SLOT_CHARACTER = ("arrival with daylight", "lecture", "discussion", "late self-study")
TOL = {"felt_c": 0.8, "lux": 110.0, "curtain": 40.0}
REACT_P = 0.5
REPEAT_MIN = 20
STUFFY_PPM = 1200.0


def campus_habits(seed: int = 2026) -> dict[tuple[str, int], dict[str, float]]:
    """Latent habit (mean preference) of every (room, slot) – fixed for the campus."""
    rng = np.random.default_rng(seed)
    out = {}
    for r in ROOM_IDS:
        off = {"setpoint": float(rng.choice([-0.5, 0.0, 0.5])), "fan": int(rng.choice([-1, 0, 1])),
               "lux": float(rng.choice([-60.0, 0.0, 60.0])), "curtain": float(rng.choice([-20.0, 0.0, 20.0])),
               "projector": float(rng.uniform(0.8, 1.1))}
        for k, base in enumerate(SLOT_BASE):
            out[(r, k)] = {
                "setpoint": base["setpoint"] + off["setpoint"],
                "fan": int(np.clip(base["fan"] + off["fan"], 1, 3)),
                "lux": base["lux"] + off["lux"],
                "curtain": float(np.clip(base["curtain"] + off["curtain"], 0, 100)),
                "projector": float(np.clip(base["projector"] * off["projector"], 0.0, 1.0)),
            }
    return out


def felt_temp(T: float, fan: int) -> float:
    return T - 0.4 * (fan - 1)


@dataclass
class Adjustment:
    day: int
    minute: int
    room: str
    slot: int
    device: str
    value: dict[str, Any]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"day": self.day, "minute": self.minute, "room": self.room, "slot": self.slot,
                "device": self.device, "value": self.value, "reason": self.reason}


class Occupants:
    """Today's occupants of all rooms (preferences = habits + daily noise)."""

    def __init__(self, habits: dict[tuple[str, int], dict[str, float]], day: int, seed: int):
        self.day = day
        self.rng = np.random.default_rng(seed)
        self.pref: dict[tuple[str, int], dict[str, Any]] = {}
        for key, h in habits.items():
            self.pref[key] = {
                "setpoint": round(h["setpoint"] + self.rng.normal(0, 0.3), 1),
                "fan": int(h["fan"]),
                "lux": h["lux"] + self.rng.normal(0, 25),
                "curtain": float(np.clip(h["curtain"] + self.rng.normal(0, 10), 0, 100)),
                "projector": bool(self.rng.random() < h["projector"]),
            }
        self._last: dict[tuple[str, str], int] = {}

    def _may(self, room: str, device: str, minute: int) -> bool:
        last = self._last.get((room, device))
        return (last is None or minute - last >= REPEAT_MIN) and self.rng.random() < REACT_P

    def react(self, rec: TickRecord, state: DeviceState, slot: int, ac_available: bool = True) -> list[Adjustment]:
        """Manual adjustments made in ``rec.room`` after experiencing this tick."""
        if rec.occupancy <= 0:
            return []
        r, m, p = rec.room, rec.minute, self.pref[(rec.room, slot)]
        out: list[Adjustment] = []

        def adjust(device: str, value: dict[str, Any], reason: str) -> None:
            if self._may(r, device, m):
                self._last[(r, device)] = m
                out.append(Adjustment(self.day, m, r, slot, device, value, reason))

        felt = felt_temp(rec.temp_c, state.ac_fan if state.ac_on() else 0)
        cur_fan = state.ac_fan if state.ac_on() else 0
        if ac_available and felt > p["setpoint"] + TOL["felt_c"]:
            # too warm: back to the usual setpoint, and never less air than now
            adjust("ac", {"setpoint": float(p["setpoint"]), "fan": int(max(p["fan"], cur_fan))}, "too warm")
        elif ac_available and felt < p["setpoint"] - TOL["felt_c"] and state.ac_on():
            adjust("ac", {"setpoint": float(p["setpoint"]), "fan": int(min(p["fan"], cur_fan))}, "too cold")
        elif ac_available and state.ac_on() and abs(state.ac_fan - p["fan"]) >= 2:
            adjust("ac", {"setpoint": float(state.ac_setpoint), "fan": int(p["fan"])}, "fan speed")
        if abs(rec.lux - p["lux"]) > TOL["lux"]:
            day_part = rec.daylight_lux * state.curtain / 100.0
            level = int(np.clip(round((p["lux"] - day_part) / E_MAX_LUX * 10.0) * 10, 0, 100))
            adjust("light", {"level": level}, "too dark" if rec.lux < p["lux"] else "too bright")
        if rec.glare:
            adjust("curtain", {"open_pct": 0}, "glare on the screen")
        elif abs(state.curtain - p["curtain"]) > TOL["curtain"]:
            adjust("curtain", {"open_pct": int(round(p["curtain"] / 10.0) * 10)}, "curtain")
        if state.projector != p["projector"]:
            adjust("projector", {"on": bool(p["projector"])}, "needs the projector" if p["projector"] else "projector not used")
        if rec.co2_ppm > STUFFY_PPM and state.vent < 3:
            adjust("vent", {"level": 3}, "stuffy")
        return out

    def discomfort(self, rec: TickRecord, state: DeviceState, slot: int) -> dict[str, bool]:
        """Which comfort criteria fail for the people in the room during this tick."""
        p = self.pref[(rec.room, slot)]
        felt = felt_temp(rec.temp_c, state.ac_fan if state.ac_on() else 0)
        return {"thermal": abs(felt - p["setpoint"]) > 1.0, "light": rec.lux < p["lux"] - 100.0,
                "co2": rec.co2_ppm > 1000.0, "glare": rec.glare}
