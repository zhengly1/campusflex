"""Room physics of the smart extension, simulated every ``TICK_MIN`` minutes.

Temperature keeps the one-zone RC model of Equation 2 (Table 6 parameters, COP 3,
q_occ 0.07 kW/person) and adds the heat of the new devices:

    C dT/dt = (T_out − T)/R + q_occ n + q_equip + q_solar + q_vent − Q_cool

* ``Q_cool`` comes from a device-local thermostat (as in the core), limited to
  COP · P_AC · health · FAN_CAPACITY[fan];
* ``q_equip`` – lights, projector and sockets turn their electrical power into heat;
* ``q_solar`` – evening sun through the windows, scaled by the curtain opening;
* ``q_vent`` – fresh air at the outdoor temperature, after a 70 % energy-recovery unit.

CO₂ follows a well-mixed mass balance (5.2·10⁻⁶ m³/s per person, outdoor 420 ppm,
0.2 air changes per hour of infiltration plus the fresh-air flow). Desk illuminance is
daylight × curtain opening + dimming level × 550 lux. All values are synthetic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..config import COP, Q_OCC_KW, ROOM_BY_ID, ROOM_IDS, START_MIN
from .devices import (
    FAN_CAPACITY,
    FAN_KW,
    PROJECTOR_KW,
    PROJECTOR_STANDBY_KW,
    SOCKET_BASE_KW,
    SOCKET_PER_PERSON_KW,
    DeviceState,
    vent_flow_m3s,
    vent_kw,
)

TICK_MIN = 2
DT_H = TICK_MIN / 60.0
E_MAX_LUX = 550.0
CO2_OUT_PPM = 420.0
CO2_GEN_M3S = 5.2e-6
INFILTRATION_ACH = 0.2
ERV_EFFICIENCY = 0.7
RHO_CP = 1.2                  # kJ/(m³·K): kW per (m³/s) per K
VOL_PER_SEAT_M3 = 5.0
SUNSET_MIN = 18 * 60 + 50
GLARE_LUX = 200.0             # daylight on the screen above this makes a projector hard to read
# Building A faces north, building B west (low evening sun).
DAYLIGHT_AT_START = {"A": 320.0, "B": 480.0}
SOLAR_KW_AT_START = {"A": 0.2, "B": 1.0}


def volume_m3(room: str) -> float:
    return ROOM_BY_ID[room].seats * VOL_PER_SEAT_M3


def _fade(minute: float, value_at_start: float, end: float) -> float:
    if minute >= end:
        return 0.0
    return value_at_start * max(0.0, (end - minute) / (end - START_MIN))


def daylight_lux(room: str, minute: float) -> float:
    """Desk illuminance from daylight with the curtain fully open."""
    return _fade(minute, DAYLIGHT_AT_START[ROOM_BY_ID[room].building], SUNSET_MIN)


def solar_kw(room: str, minute: float) -> float:
    """Solar heat gain with the curtain fully open."""
    return _fade(minute, SOLAR_KW_AT_START[ROOM_BY_ID[room].building], SUNSET_MIN - 10)


@dataclass
class RoomTruth:
    temp_c: float
    co2_ppm: float = 450.0


@dataclass
class TickRecord:
    """What physically happened in one room during one tick."""

    room: str
    minute: int
    occupancy: int
    temp_c: float
    co2_ppm: float
    lux: float
    daylight_lux: float
    glare: bool
    cooling_kw: float
    power_kw: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"room": self.room, "minute": self.minute, "occupancy": self.occupancy,
                "temp_c": round(self.temp_c, 3), "co2_ppm": round(self.co2_ppm, 1), "lux": round(self.lux, 1),
                "daylight_lux": round(self.daylight_lux, 1), "glare": self.glare,
                "cooling_kw": round(self.cooling_kw, 3), "power_kw": {k: round(v, 4) for k, v in self.power_kw.items()}}


class SmartPlant:
    """Physical state of the six rooms and their devices."""

    def __init__(self) -> None:
        self.truth = {r: RoomTruth(ROOM_BY_ID[r].T0) for r in ROOM_IDS}
        self.devices = {r: DeviceState() for r in ROOM_IDS}
        self.health = {r: 1.0 for r in ROOM_IDS}

    def step(self, minute: int, t_out: float, occupancy: dict[str, int]) -> dict[str, TickRecord]:
        """Advance every room by one tick under the applied device states."""
        out = {}
        for r in ROOM_IDS:
            spec, s, x = ROOM_BY_ID[r], self.devices[r], self.truth[r]
            n = int(occupancy.get(r, 0))
            day = daylight_lux(r, minute)
            open_f = s.curtain / 100.0
            p = {
                "light": spec.P_light * s.light / 100.0,
                "vent": vent_kw(r, s.vent),
                "projector": PROJECTOR_KW if s.projector else PROJECTOR_STANDBY_KW,
                "sockets": (SOCKET_BASE_KW + SOCKET_PER_PERSON_KW * n) if s.sockets else 0.0,
            }
            flow = vent_flow_m3s(r, s.vent)
            q_vent = RHO_CP * flow * (t_out - x.temp_c) * (1.0 - ERV_EFFICIENCY)
            load = ((t_out - x.temp_c) / spec.R + Q_OCC_KW * n + p["light"] + p["projector"] + p["sockets"]
                    + solar_kw(r, minute) * open_f + q_vent)
            cool = 0.0
            if s.ac_on():
                cap = COP * spec.P_ac * self.health[r] * FAN_CAPACITY[s.ac_fan]
                need = load + (x.temp_c - float(s.ac_setpoint)) * spec.C / DT_H
                cool = min(max(need, 0.0), cap)
            p["ac"] = (cool / COP + FAN_KW[s.ac_fan]) if s.ac_on() else 0.0
            x.temp_c = x.temp_c + DT_H / spec.C * (load - cool)
            vol = volume_m3(r)
            q = flow + vol * INFILTRATION_ACH / 3600.0
            c_eq = CO2_OUT_PPM + n * CO2_GEN_M3S * 1e6 / q
            x.co2_ppm = c_eq + (x.co2_ppm - c_eq) * math.exp(-q / vol * TICK_MIN * 60.0)
            lux = day * open_f + s.light / 100.0 * E_MAX_LUX
            out[r] = TickRecord(r, minute, n, x.temp_c, x.co2_ppm, lux, day,
                                glare=s.projector and day * open_f > GLARE_LUX, cooling_kw=cool, power_kw=p)
        return out
