"""Room sensors: noisy, quantized, occasionally missing, and with injectable faults.

Each room reports, every tick: temperature, CO₂, desk illuminance, a people counter, a
motion (PIR) flag and one power meter per device. Faults are scheduled per day
(``SensorFault``) and kept as ground truth so that anomaly detection can be scored:

* ``stuck`` – the reading freezes at its value when the fault starts;
* ``drift`` – the reading drifts by ``magnitude`` units per minute;
* ``spike`` – a single reading is off by ``magnitude``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..config import END_MIN
from .physics import TICK_MIN, TickRecord

NOISE = {"temp": 0.15, "co2": 25.0, "lux": 8.0}
QUANT = {"temp": 0.1, "co2": 5.0, "lux": 1.0}
FIELD = {"temp": "temp_c", "co2": "co2_ppm", "lux": "lux"}


def noise_sd(sensor: str, value: float) -> float:
    """Measurement noise (1 σ); illuminance sensors are worse in proportion to the reading."""
    return NOISE[sensor] + (0.03 * max(0.0, value) if sensor == "lux" else 0.0)


@dataclass(frozen=True)
class SensorFault:
    room: str
    sensor: str              # temp | co2 | lux
    kind: str                # stuck | drift | spike
    start_min: int
    end_min: int = END_MIN
    magnitude: float = 0.0

    def active(self, minute: int) -> bool:
        if self.kind == "spike":
            return self.start_min <= minute < self.start_min + TICK_MIN
        return self.start_min <= minute < self.end_min

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SensorSuite:
    def __init__(self, seed: int, faults: tuple[SensorFault, ...] | list[SensorFault] = (), dropout: float = 0.01):
        self.rng = np.random.default_rng(seed)
        self.faults = list(faults)
        self.dropout = dropout
        self._frozen: dict[tuple[str, str], float] = {}

    def fault_at(self, room: str, sensor: str, minute: int) -> SensorFault | None:
        return next((f for f in self.faults if f.room == room and f.sensor == sensor and f.active(minute)), None)

    def _measure(self, room: str, sensor: str, true_value: float, minute: int) -> float | None:
        f = self.fault_at(room, sensor, minute)
        if f is not None and f.kind == "stuck":
            key = (room, sensor)
            if key not in self._frozen:
                self._frozen[key] = self._quantize(sensor, true_value + self.rng.normal(0, NOISE[sensor]))
            return self._frozen[key]
        if self.rng.random() < self.dropout:
            return None
        v = true_value + self.rng.normal(0, noise_sd(sensor, true_value))
        if f is not None and f.kind == "drift":
            v += f.magnitude * (minute - f.start_min)
        elif f is not None and f.kind == "spike":
            v += f.magnitude
        return self._quantize(sensor, max(0.0, v) if sensor != "temp" else v)

    @staticmethod
    def _quantize(sensor: str, v: float) -> float:
        q = QUANT[sensor]
        return round(round(v / q) * q, 3)

    def read(self, rec: TickRecord) -> dict[str, Any]:
        """One telemetry message for ``rec``'s room."""
        n = rec.occupancy
        counted = n
        if n > 0 and self.rng.random() < 0.1:
            counted = max(0, n + int(self.rng.choice([-2, -1, 1, 2])))
        motion = (n > 0 and self.rng.random() < 0.97) or (n == 0 and self.rng.random() < 0.01)
        return {
            "room": rec.room,
            "minute": rec.minute,
            "temp_c": self._measure(rec.room, "temp", rec.temp_c, rec.minute),
            "co2_ppm": self._measure(rec.room, "co2", rec.co2_ppm, rec.minute),
            "lux": self._measure(rec.room, "lux", rec.lux, rec.minute),
            "occupancy": counted,
            "motion": bool(motion),
            "power_kw": {k: round(v, 4) for k, v in rec.power_kw.items()},
        }
