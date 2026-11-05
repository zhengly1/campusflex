"""Versioned state snapshot returned by ``get_state`` (tool contract, Section 3.5)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import N_ROOMS, ROOM_IDS, ROOMS, clock_str
from .scenarios import Notice


@dataclass
class Snapshot:
    version: int
    step: int
    clock_min: int
    demand: int
    t_out: float
    temps: np.ndarray            # observed room temperatures (°C)
    health: np.ndarray           # AC health 0..1
    open: np.ndarray             # bool
    counts: np.ndarray           # seated students per room
    capacity: np.ndarray         # seats (0 when under maintenance)
    steps_since_switch: np.ndarray
    movable: np.ndarray          # students per room eligible to be moved (dwell, max moves, windows)
    notices: list[Notice] = field(default_factory=list)
    stale: np.ndarray = field(default_factory=lambda: np.zeros(N_ROOMS, dtype=bool))
    rechecks: list[str] = field(default_factory=list)   # rooms flagged for sensor recheck
    energy_used_kwh: float = 0.0                        # metered energy so far this evening

    @property
    def seated(self) -> int:
        return int(self.counts.sum())

    @property
    def waiting(self) -> int:
        return max(0, self.demand - self.seated)

    @property
    def open_set(self) -> frozenset[str]:
        return frozenset(r for r, o in zip(ROOM_IDS, self.open) if o)

    def summary(self) -> dict[str, Any]:
        """Compact state summary handed to LLM roles (numbers come from tools, not the LLM)."""
        rooms = []
        for i, r in enumerate(ROOMS):
            rooms.append(
                {
                    "room": r.room_id,
                    "seats": int(self.capacity[i]),
                    "open": bool(self.open[i]),
                    "occupants": int(self.counts[i]),
                    "temp_c": round(float(self.temps[i]), 2),
                    "ac_ok": bool(self.health[i] > 0),
                    "sensor_stale": bool(self.stale[i]),
                    "movable": int(self.movable[i]),
                }
            )
        return {
            "time": clock_str(self.clock_min),
            "state_version": self.version,
            "demand": int(self.demand),
            "waiting": self.waiting,
            "outdoor_c": round(float(self.t_out), 2),
            "rooms": rooms,
        }
