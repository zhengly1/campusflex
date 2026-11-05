"""Declared constants and configuration (PDF Sections 2.1, 3.1, Tables 4, 6, 8, 9).

Every value here is a *declared simulation setting* of the synthetic six-room campus,
not an institutional standard or a measured campus value.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

# --- Time grid ---------------------------------------------------------------
STEP_MIN = 10                      # decision every ten minutes
DT_H = STEP_MIN / 60.0             # Δt = 1/6 h
START_MIN = 18 * 60                # 18:00
END_MIN = 22 * 60                  # 22:00
N_STEPS = (END_MIN - START_MIN) // STEP_MIN   # 24 decision steps

# --- Thermal constants (Eq. 2) ----------------------------------------------
Q_OCC_KW = 0.07                    # kW per person
COP = 3.0

# --- Comfort settings (declared, Section 2.1) --------------------------------
BAND_LOW_C = 24.0
BAND_HIGH_C = 27.0
TARGET_C = 26.0
TRANSFER_C = 30.0                  # occupants must be transferred above this

# Reference outdoor temperature used only to rank rooms by efficiency.
EFFICIENCY_REF_TOUT_C = 30.0


@dataclass(frozen=True)
class RoomSpec:
    room_id: str
    building: str
    seats: int
    R: float        # °C/kW
    C: float        # kWh/°C
    P_ac: float     # kW electrical
    P_light: float  # kW
    T0: float       # °C initial temperature


# Table 4 + Table 6
ROOMS: tuple[RoomSpec, ...] = (
    RoomSpec("A101", "A", 60, 1.2, 4.5, 4.0, 0.65, 26.4),
    RoomSpec("A102", "A", 50, 0.8, 4.0, 4.5, 0.65, 27.0),
    RoomSpec("A103", "A", 40, 0.9, 3.2, 3.6, 0.50, 26.8),
    RoomSpec("B201", "B", 70, 1.8, 5.0, 4.2, 0.60, 26.2),
    RoomSpec("B202", "B", 45, 1.5, 3.5, 3.2, 0.40, 27.0),
    RoomSpec("B203", "B", 35, 1.7, 2.8, 2.6, 0.35, 26.5),
)
ROOM_IDS: tuple[str, ...] = tuple(r.room_id for r in ROOMS)
ROOM_INDEX: dict[str, int] = {r: i for i, r in enumerate(ROOM_IDS)}
ROOM_BY_ID: dict[str, RoomSpec] = {r.room_id: r for r in ROOMS}
N_ROOMS = len(ROOMS)
TOTAL_SEATS = sum(r.seats for r in ROOMS)

# Table 8: declared weight levels for the intent compiler.
WEIGHT_LEVELS: dict[str, dict[str, float]] = {
    "lambda_M": {"low": 0.02, "default": 0.06, "high": 0.20},
    "lambda_T": {"low": 4.0, "default": 8.0, "high": 16.0},
    "lambda_S": {"low": 0.04, "default": 0.08, "high": 0.16},
}
LEVEL_ORDER = ("low", "default", "high")

# Seed protocol (Section 4.2, L5 acceptance rule, E6)
DEV_SEEDS = tuple(range(0, 10))
TUNING_SEEDS = tuple(range(0, 7))
CHECK_SEEDS = tuple(range(7, 10))
HELDOUT_SEEDS = tuple(range(10, 20))
HELDOUT2_SEEDS = tuple(range(20, 30))

# Table 9: L5 diff space. Ordered lists; "one step" means an adjacent value.
L5_PARAM_SPACE: dict[str, list[Any]] = {
    "lambda_T": [4.0, 8.0, 16.0],
    "lambda_M": [0.02, 0.06, 0.20],
    "lambda_S": [0.04, 0.08, 0.16],
    "lambda_Omega": [0.0, 2.0, 4.0, 8.0],
    "rollout_horizon_min": [40, 60, 80],
    "l2_temp_trigger_c": [26.5, 26.8, 27.0],
    "l2_periodic_every": [2, 3, 4],
    "precool_set_min": [[10, 20], [10, 20, 30], [20, 30, 40]],
}


def clock_str(minute_of_day: int) -> str:
    return f"{minute_of_day // 60:02d}:{minute_of_day % 60:02d}"


def parse_clock(s: str) -> int:
    hh, mm = s.strip().split(":")
    return int(hh) * 60 + int(mm)


def step_to_clock(step: int) -> int:
    return START_MIN + step * STEP_MIN


def clock_to_step(minute_of_day: int) -> int:
    return (minute_of_day - START_MIN) // STEP_MIN


@dataclass
class PlannerConfig:
    """Tunable controller configuration. Hard constraints are NOT here (see validator)."""

    # Equation 4 weights. v1: λT=8, λM=0.06, λS=0.08, no terminal term.
    lambda_T: float = 8.0
    lambda_M: float = 0.06
    lambda_S: float = 0.08
    lambda_Omega: float = 4.0          # declared; tuned on development seeds only
    rollout_horizon_min: int = 60      # v2 rollout length
    omega_window_min: int = 30         # Ω window after the rollout ends

    # L2 triggers (Section 3.4)
    l2_temp_trigger_c: float = 26.8
    l2_periodic_every: int = 3
    l2_demand_trigger: int = 20
    l2_max_plans: int = 5

    # Rule-staged generator (Section 3.3)
    precool_set_min: list[int] = field(default_factory=lambda: [10, 20, 30])
    rule_staged_temp_c: float = 26.8

    # Relocation / dwell policy (L1 can tighten these)
    min_room_dwell_steps: int = 2
    min_student_dwell_steps: int = 3
    max_moves_per_student: int = 2
    no_move_windows: list[list[str]] = field(default_factory=list)   # [["20:00","22:00"]]

    # Optional request limits compiled by L1
    energy_budget_kwh: float | None = None
    power_cap_kw: float | None = None
    band_low_c: float = BAND_LOW_C
    band_high_c: float = BAND_HIGH_C
    target_c: float = TARGET_C

    # Fixed-opening baseline plans for this many seats (inventory order).
    fixed_planned_seats: int = 180

    # L3 forecast adjustment bounds
    forecast_delta_cap: int = 80
    weather_delta_cap_c: float = 5.0

    # LLM guardrails (Section 3.4 "Common guardrails")
    llm_calls_per_step: int = 3
    llm_latency_s: float = 20.0

    # L4 trade-off trigger
    tradeoff_dE_kwh: float = 1.0
    tradeoff_dDev_c: float = 0.02

    # Departures: "random" (v2) or "highest_id" (v1 bias)
    departure_mode: str = "random"

    @property
    def horizon_steps(self) -> int:
        return max(1, self.rollout_horizon_min // STEP_MIN)

    @property
    def omega_steps(self) -> int:
        return max(0, self.omega_window_min // STEP_MIN)

    @classmethod
    def v1(cls) -> "PlannerConfig":
        """v1 settings: single 30-minute mask hold, no terminal term, highest-id departures."""
        return cls(lambda_Omega=0.0, rollout_horizon_min=30, departure_mode="highest_id")

    def with_levels(self, levels: dict[str, str]) -> "PlannerConfig":
        upd = {k: WEIGHT_LEVELS[k][v] for k, v in levels.items() if k in WEIGHT_LEVELS}
        return replace(self, **upd)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PlannerConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "PlannerConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))
