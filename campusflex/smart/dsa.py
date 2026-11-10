"""Sensor data processing for fine-grained control (DSA).

Per room, every tick:

1. **Prediction** of each reading from a model: temperature from an online-identified
   thermal model (recursive least squares, ``ThermalRLS``), CO₂ from the mass balance with
   the known fresh-air level and the counted occupants, illuminance from the dimming level
   plus the estimated daylight.
2. **Anomaly detection** on the innovation (reading − prediction): a reading more than five
   standard deviations off is a *spike* and is dropped; identical readings while the model
   expects change mean a *stuck* sensor; a two-sided CUSUM on the innovations detects a
   *drift*. A flagged sensor is ignored and the model prediction is used instead (soft
   sensor) until its readings agree with the model again.
3. **Filtering** of the accepted readings with a scalar Kalman filter.

``ThermalRLS`` fits ΔT = a·q_known + b·(T_out − T), where q_known is the heat the
system knows (occupants at 0.07 kW each, the metered power of lights, projector and sockets,
fresh air after heat recovery, minus the cooling derived from the AC power meter). Then
C = Δt / a and R = a / b (``identified``); the scheduler uses them for pre-cooling lead times
and the experiment compares them with Table 6. Holding the setpoint, the AC balances the load,
which pins down R; pull-downs and switching pin down C.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import COP, Q_OCC_KW, ROOM_BY_ID
from .devices import FAN_CAPACITY, FAN_KW, DeviceState, vent_flow_m3s
from .physics import (
    CO2_GEN_M3S,
    CO2_OUT_PPM,
    DT_H,
    E_MAX_LUX,
    ERV_EFFICIENCY,
    INFILTRATION_ACH,
    RHO_CP,
    TICK_MIN,
    volume_m3,
)
from .sensors import FIELD, NOISE, noise_sd

SENSORS = ("temp", "co2", "lux")


class Kalman1D:
    def __init__(self, x0: float, p0: float, q: float, r: float):
        self.x, self.p, self.q, self.r = float(x0), float(p0), float(q), float(r)

    def predict(self, x_pred: float | None = None, q: float | None = None, r: float | None = None) -> None:
        if x_pred is not None:
            self.x = float(x_pred)
        self.p += self.q if q is None else q
        if r is not None:
            self.r = float(r)

    def z_score(self, z: float) -> float:
        return (z - self.x) / math.sqrt(self.p + self.r)

    def update(self, z: float) -> None:
        k = self.p / (self.p + self.r)
        self.x += k * (z - self.x)
        self.p *= 1.0 - k


class ThermalRLS:
    """Recursive least squares with forgetting for the per-tick temperature change."""

    N = 2

    def __init__(self, lam: float = 0.998, delta: float = 50.0):
        self.lam = lam
        self.theta = np.zeros(self.N)
        self.P = np.eye(self.N) * delta
        self.n = 0

    @staticmethod
    def features(T: float, t_out: float, n: float, cool_kw: float, equip_kw: float, flow: float) -> np.ndarray:
        known = Q_OCC_KW * n + equip_kw + RHO_CP * (1.0 - ERV_EFFICIENCY) * flow * (t_out - T) - cool_kw
        # no intercept: a constant term absorbs the slow outdoor drift that identifies R
        return np.array([known, t_out - T])

    def update(self, x: np.ndarray, dT: float) -> None:
        Px = self.P @ x
        k = Px / (self.lam + x @ Px)
        self.theta = self.theta + k * (dT - x @ self.theta)
        self.P = (self.P - np.outer(k, Px)) / self.lam
        self.n += 1

    def predict(self, x: np.ndarray) -> float:
        return float(x @ self.theta)

    def ready(self, min_samples: int = 30) -> bool:
        return self.n >= min_samples and self.theta[0] > 0 and self.theta[1] > 0

    def identified(self) -> dict[str, float] | None:
        """R (°C/kW) and C (kWh/°C) implied by the coefficients, once the model is usable."""
        if not self.ready():
            return None
        return {"R": float(self.theta[0] / self.theta[1]), "C": float(DT_H / self.theta[0])}


@dataclass
class SensorMonitor:
    """Spike / stuck / drift detection for one sensor.

    ``residual`` is the reading minus an *independent* reference (not the Kalman estimate,
    which follows a drifting sensor): a stuck reading repeats exactly although real readings
    are noisy; a drift accumulates in a two-sided CUSUM of the standardized residual; a
    single reading more than five standard deviations off is a spike and is dropped."""

    sensor: str
    window: int = 6
    cusum_k: float = 1.0
    cusum_h: float = 10.0
    flagged: str | None = None           # None | stuck | drift
    flagged_at: int | None = None
    recent: deque = field(default_factory=lambda: deque(maxlen=8))
    s_pos: float = 0.0
    s_neg: float = 0.0
    calm: int = 0

    WINDOW = {"temp": 8, "co2": 6, "lux": 6}
    DRIFT = {"temp": True, "co2": True, "lux": False}     # daylight makes the lux reference too loose

    def check(self, z: float | None, ref: float, sd: float, minute: int, ref_ok: bool = True) -> str:
        """``ok`` | ``missing`` | ``spike`` | ``stuck`` | ``drift`` – the verdict for this reading.

        ``ref_ok`` is False while the reference cannot be trusted (e.g. before the thermal model
        is identified, when only spikes are screened)."""
        if z is None:
            return "missing" if self.flagged is None else self.flagged
        self.recent.append(z)
        w = self.WINDOW[self.sensor]
        tail = list(self.recent)[-w:]
        stuck = len(tail) == w and len(set(tail)) == 1 and (self.sensor != "lux" or ref > 50.0)
        if stuck:
            return self._flag("stuck", minute)
        zr = (z - ref) / sd
        if self.flagged is not None:
            # recovery: readings vary again and agree with the reference for ten ticks
            self.calm = self.calm + 1 if (abs(zr) < 2.0 and len(set(tail)) > 1) else 0
            if self.calm >= 10:
                self.flagged, self.flagged_at, self.calm, self.s_pos, self.s_neg = None, None, 0, 0.0, 0.0
                return "ok"
            return self.flagged
        if ref_ok and self.DRIFT[self.sensor]:
            self.s_pos = max(0.0, self.s_pos + zr - self.cusum_k)
            self.s_neg = max(0.0, self.s_neg - zr - self.cusum_k)
            if self.s_pos > self.cusum_h or self.s_neg > self.cusum_h:
                return self._flag("drift", minute)
        if abs(zr) > (5.0 if ref_ok else 8.0):
            return "spike"
        return "ok"

    def _flag(self, kind: str, minute: int) -> str:
        if self.flagged is None:
            self.flagged, self.flagged_at = kind, minute
        self.calm = 0
        return self.flagged


@dataclass
class Estimate:
    room: str
    minute: int
    temp_c: float
    co2_ppm: float
    lux: float
    daylight_lux: float
    occupancy: int
    verdicts: dict[str, str]
    faulty: dict[str, str]
    identified: dict[str, float] | None = None      # R, C of the room from the thermal model

    def to_dict(self) -> dict[str, Any]:
        return {"room": self.room, "minute": self.minute, "temp_c": round(self.temp_c, 3),
                "co2_ppm": round(self.co2_ppm, 1), "lux": round(self.lux, 1),
                "daylight_lux": round(self.daylight_lux, 1), "occupancy": self.occupancy,
                "verdicts": self.verdicts, "faulty": self.faulty}


def co2_after(c0: float, n: float, vent_level: int, room: str, minutes: float) -> float:
    """Mass-balance CO₂ after ``minutes`` with ``n`` occupants and the given fresh-air level."""
    vol = volume_m3(room)
    q = vent_flow_m3s(room, vent_level) + vol * INFILTRATION_ACH / 3600.0
    c_eq = CO2_OUT_PPM + n * CO2_GEN_M3S * 1e6 / q
    return c_eq + (c0 - c_eq) * math.exp(-q / vol * minutes * 60.0)


class RoomEstimator:
    """Model-based estimation and anomaly detection for one room's sensors.

    References used to judge each sensor (independent of that sensor's own history):

    * CO₂ – an open-loop mass balance driven by the people counter and the fresh-air level,
      pulled only slowly (2 % per tick) toward accepted readings;
    * temperature – physical redundancy: while the AC runs below its capacity (from its power
      meter) the thermostat holds the room at the setpoint, and only then is a drift called;
      otherwise the identified thermal model propagates the estimate;
    * illuminance – dimming level × 550 lux plus the estimated daylight.
    """

    def __init__(self, room: str, rls: ThermalRLS | None = None):
        self.room = room
        T0 = ROOM_BY_ID[room].T0
        self.kf = {"temp": Kalman1D(T0, 0.5, 0.01, NOISE["temp"] ** 2),
                   "co2": Kalman1D(450.0, 400.0, 30.0 ** 2, NOISE["co2"] ** 2),
                   "lux": Kalman1D(0.0, 400.0, 20.0 ** 2, NOISE["lux"] ** 2)}
        self.mon = {s: SensorMonitor(s) for s in SENSORS}
        self.rls = rls if rls is not None else ThermalRLS()       # shared across evenings: the building is learned
        self.daylight = 0.0
        self.occupancy = 0
        self.ref = {"temp": T0, "co2": 450.0}
        self._last_T: float | None = None
        self.flags: list[dict[str, Any]] = []

    def update(self, msg: dict[str, Any], t_out: float, state: DeviceState) -> Estimate:
        """Fuse one telemetry message; ``state`` is the device state applied during the tick."""
        minute = int(msg["minute"])
        spec = ROOM_BY_ID[self.room]
        if msg.get("occupancy") is not None:
            self.occupancy = int(msg["occupancy"])
        n = self.occupancy
        power = msg.get("power_kw") or {}
        ac_kw = float(power.get("ac", 0.0))
        fan_kw = FAN_KW[state.ac_fan] if state.ac_on() else 0.0
        cool = max(0.0, COP * (ac_kw - fan_kw))
        equip = float(power.get("light", 0.0)) + float(power.get("projector", 0.0)) + float(power.get("sockets", 0.0))
        flow = vent_flow_m3s(self.room, state.vent)
        # ---- independent references
        x_ref = ThermalRLS.features(self.ref["temp"], t_out, n, cool, equip, flow)
        holding = state.ac_on() and ac_kw - fan_kw < 0.97 * spec.P_ac * FAN_CAPACITY[state.ac_fan]
        if holding and cool > 0:
            self.ref["temp"] = float(state.ac_setpoint)
        elif self.rls.ready():
            self.ref["temp"] += self.rls.predict(x_ref)
        else:
            self.ref["temp"] = self.kf["temp"].x
        temp_ref_ok = (holding and cool > 0) or self.rls.ready()
        self.ref["co2"] = co2_after(self.ref["co2"], n, state.vent, self.room, TICK_MIN)
        lux_ref = state.light / 100.0 * E_MAX_LUX + self.daylight * state.curtain / 100.0
        refs = {"temp": self.ref["temp"], "co2": self.ref["co2"], "lux": lux_ref}
        sds = {"temp": math.hypot(NOISE["temp"], 0.15), "co2": math.hypot(NOISE["co2"], 30.0),
               "lux": noise_sd("lux", lux_ref) + 40.0}
        # ---- Kalman prediction (model step from the filtered state)
        T_prev = self.kf["temp"].x
        x = ThermalRLS.features(T_prev, t_out, n, cool, equip, flow)
        preds = {"temp": (float(state.ac_setpoint) if (holding and cool > 0) else
                          T_prev + self.rls.predict(x) if self.rls.ready() else T_prev),
                 "co2": co2_after(self.kf["co2"].x, n, state.vent, self.room, TICK_MIN),
                 "lux": lux_ref}
        verdicts, faulty = {}, {}
        for s in SENSORS:
            kf, z = self.kf[s], msg.get(FIELD[s])
            q = None if s != "temp" or temp_ref_ok else 0.05
            kf.predict(preds[s], q=q, r=noise_sd(s, preds[s]) ** 2)
            # temperature drift is judged only against the thermostat redundancy (AC holding the
            # setpoint); a free-floating or saturated room is propagated by the model, which is
            # good enough for estimation but too loose to call a sensor faulty
            v = self.mon[s].check(z, refs[s], sds[s], minute, ref_ok=bool(holding and cool > 0) if s == "temp" else True)
            verdicts[s] = v
            if v == "ok":
                kf.update(z)
                if s == "co2":
                    self.ref["co2"] += 0.02 * (z - self.ref["co2"])
            elif v in ("stuck", "drift"):
                faulty[s] = v
                if self.mon[s].flagged_at == minute:
                    self.flags.append({"room": self.room, "sensor": s, "kind": v, "minute": minute})
        # daylight = what the lux sensor sees beyond the artificial light (only with the curtain open)
        if verdicts["lux"] == "ok" and state.curtain > 0:
            seen = max(0.0, self.kf["lux"].x - state.light / 100.0 * E_MAX_LUX) / (state.curtain / 100.0)
            self.daylight = 0.7 * self.daylight + 0.3 * seen
        else:
            self.daylight *= 0.9            # it is getting dark; decay toward zero without evidence
        # identify the thermal model from the change this tick between two accepted raw readings
        T_now = self.kf["temp"].x
        z_temp = msg.get("temp_c")
        if verdicts["temp"] == "ok" and self._last_T is not None:
            self.rls.update(x, float(z_temp) - self._last_T)
        self._last_T = float(z_temp) if verdicts["temp"] == "ok" else None
        if verdicts["temp"] == "ok" and not temp_ref_ok:
            self.ref["temp"] = T_now
        return Estimate(self.room, minute, T_now, self.kf["co2"].x, self.kf["lux"].x, self.daylight, n,
                        verdicts, faulty, self.rls.identified())


def score_detection(flags: list[dict[str, Any]], faults: list[Any], window_min: int = 30) -> dict[str, Any]:
    """Detected stuck/drift faults vs injected ones (spikes are dropped, not flagged).

    A fault counts as detected when a flag on the same room and sensor is raised within
    ``window_min`` of its start; a flag that matches no fault is a false alarm."""
    target = [f for f in faults if f.kind in ("stuck", "drift")]
    hit, used = 0, set()
    delays = []
    for f in target:
        m = next((i for i, g in enumerate(flags) if i not in used and g["room"] == f.room and g["sensor"] == f.sensor
                  and f.start_min <= g["minute"] <= f.start_min + window_min), None)
        if m is not None:
            used.add(m)
            hit += 1
            delays.append(flags[m]["minute"] - f.start_min)
    false_alarms = len(flags) - len(used)
    return {"faults": len(target), "detected": hit, "false_alarms": false_alarms,
            "recall": hit / len(target) if target else None,
            "precision": hit / len(flags) if flags else None,
            "mean_delay_min": float(np.mean(delays)) if delays else None}

