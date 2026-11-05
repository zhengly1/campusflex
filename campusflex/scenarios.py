"""Synthetic scenarios (Section 2.1, 4.1, 4.2).

v1 family: steady, surge, heat (surge+heat), fault (surge+heat+B201 fault at 20:00).
v2 adds: fault_surge (B201 offline at 19:30 during the surge), double_fault (B201 and A102
offline from 20:00) and a Notice family in which demand changes are announced in text
10-40 minutes before they happen.

All demand, weather and fault values are synthetic. Seeds make every scenario reproducible.
"""
from __future__ import annotations

import json
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    N_ROOMS,
    N_STEPS,
    ROOM_IDS,
    ROOM_INDEX,
    START_MIN,
    STEP_MIN,
    clock_str,
    clock_to_step,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

V1_SCENARIOS = ("steady", "surge", "heat", "fault")
E1_SCENARIOS = ("heat", "fault", "fault_surge", "double_fault")
NOTICE_SCENARIOS = ("notice_surge", "notice_maintenance", "notice_drop", "notice_fault")
ALL_SCENARIOS = V1_SCENARIOS + ("fault_surge", "double_fault") + NOTICE_SCENARIOS


@dataclass
class Notice:
    """A text notice or occupant report. ``truth`` is the oracle adjustment (None for distractors)."""

    notice_id: str
    release_min: int                 # minute of day at which the operator receives it
    text: str
    kind: str = "notice"             # "notice" | "report"
    truth: dict[str, Any] | None = None
    distractor: bool = False
    injected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "notice_id": self.notice_id,
            "release": clock_str(self.release_min),
            "text": self.text,
            "kind": self.kind,
            "truth": self.truth,
            "distractor": self.distractor,
            "injected": self.injected,
        }


@dataclass
class Scenario:
    name: str
    seed: int
    demand: np.ndarray                     # int, (N_STEPS,) students wanting a seat at each step
    t_out: np.ndarray                      # float, (N_STEPS + 1,)
    health: np.ndarray                     # float, (N_STEPS, N_ROOMS) AC health h_{i,t}
    unavailable: frozenset[str] = frozenset()   # rooms under maintenance (capacity 0)
    notices: list[Notice] = field(default_factory=list)
    mismatch: dict[str, dict[str, float]] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def family(self) -> str:
        return "notice" if self.name.startswith("notice") else "core"

    def capacities(self) -> np.ndarray:
        from .config import ROOMS

        return np.array([0 if r.room_id in self.unavailable else r.seats for r in ROOMS], dtype=int)

    def released_notices(self, minute_of_day: int) -> list[Notice]:
        return [n for n in self.notices if n.release_min <= minute_of_day]

    def without_notice(self, notice_id: str) -> "Scenario":
        import copy

        sc = copy.deepcopy(self)
        sc.notices = [n for n in sc.notices if n.notice_id != notice_id]
        return sc

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "unavailable": sorted(self.unavailable),
            "mismatch": self.mismatch is not None,
            "events": self.events,
            "notices": [n.to_dict() for n in self.notices],
        }


def _rng(name: str, seed: int, salt: str = "") -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(f"{name}|{salt}".encode())])


def _load_corpus() -> dict[str, Any]:
    p = DATA_DIR / "notice_corpus.json"
    return json.loads(p.read_text())


@dataclass
class ScenarioInputs:
    """Scenario inputs the PDF leaves open (weather, demand shape, timing), in one place.

    Anchored by the PDF: 120 students at 18:00 rising toward 180 (Section 2.1), heat in the
    heat and fault scenarios, B201's AC failing at 20:00. The other values are this project's
    own settings; their fingerprint is part of the freeze record (``scenario_hash``)."""

    tout_start_c: float = 29.0          # outdoor temperature at 18:00 ...
    tout_end_c: float = 27.0            # ... falling linearly to 22:00
    tout_jitter_c: float = 0.5          # per-seed offset, uniform ±
    heat_rise_c: float = 3.0            # heat scenarios: rise of the outdoor temperature
    heat_start_min: int = 19 * 60       # earliest start of the rise (+0-30 min per seed)
    heat_ramp_steps: int = 2
    steady_demand: int = 120
    surge_base: int = 120
    surge_peak: int = 180
    surge_late: int = 90                # demand after the surge ends
    surge_start_min: int = 18 * 60 + 30  # surge starts here (+0-10 min per seed)
    surge_ramp_steps: int = 3
    surge_end_min: int = 21 * 60


INPUTS = ScenarioInputs()


def _base_tout(rng: np.random.Generator) -> np.ndarray:
    off = rng.uniform(-INPUTS.tout_jitter_c, INPUTS.tout_jitter_c)
    return np.linspace(INPUTS.tout_start_c, INPUTS.tout_end_c, N_STEPS + 1) + off


def _add_heat(t_out: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    start = clock_to_step(INPUTS.heat_start_min) + int(rng.integers(0, 4))
    rise = INPUTS.heat_rise_c + rng.uniform(-0.3, 0.3)
    t = t_out.copy()
    for k in range(start, N_STEPS + 1):
        t[k] += rise * min(1.0, (k - start + 1) / float(INPUTS.heat_ramp_steps))
    return t, start


def _steady_demand(rng: np.random.Generator) -> np.ndarray:
    return np.full(N_STEPS, INPUTS.steady_demand + int(rng.integers(-5, 6)), dtype=int)


def _surge_demand(rng: np.random.Generator) -> np.ndarray:
    base = INPUTS.surge_base + int(rng.integers(-4, 5))
    peak = INPUTS.surge_peak + int(rng.integers(-6, 7))
    late = INPUTS.surge_late + int(rng.integers(-5, 6))
    s0 = clock_to_step(INPUTS.surge_start_min) + int(rng.integers(0, 2))
    ramp = INPUTS.surge_ramp_steps
    d = np.empty(N_STEPS, dtype=int)
    for k in range(N_STEPS):
        if k < s0:
            d[k] = base
        elif k < s0 + ramp:
            d[k] = round(base + (peak - base) * (k - s0 + 1) / ramp)
        elif k < clock_to_step(INPUTS.surge_end_min):
            d[k] = peak
        else:
            d[k] = late
    return d


def _offline(health: np.ndarray, room: str, from_min: int) -> None:
    health[clock_to_step(from_min):, ROOM_INDEX[room]] = 0.0


def _fmt(tpl: str, **kw: Any) -> str:
    return tpl.format(**kw)


def _pick(rng: np.random.Generator, seq: list[str]) -> str:
    return seq[int(rng.integers(0, len(seq)))]


def _demand_notice(
    rng: np.random.Generator,
    corpus: dict[str, Any],
    wording: str,
    nid: str,
    event_min: int,
    end_min: int,
    announced: int,
    up: bool,
) -> Notice:
    lead = int(rng.choice([10, 20, 30, 40]))
    tpl = _pick(rng, corpus[wording]["demand_up" if up else "demand_down"])
    text = _fmt(tpl, place=_pick(rng, corpus["places"]), start=clock_str(event_min), n=announced)
    return Notice(
        notice_id=nid,
        release_min=event_min - lead,
        text=text,
        truth={
            "kind": "demand",
            "start": clock_str(event_min),
            "end": clock_str(end_min),
            "delta": announced if up else -announced,
        },
    )


def _distractors(rng: np.random.Generator, corpus: dict[str, Any], wording: str, start_idx: int, k: int) -> list[Notice]:
    out = []
    for j in range(k):
        tpl = _pick(rng, corpus[wording]["distractor"])
        out.append(
            Notice(
                notice_id=f"N{start_idx + j}",
                release_min=START_MIN + 10 * int(rng.integers(0, 18)),
                text=_fmt(tpl, building=_pick(rng, ["A", "B"])),
                distractor=True,
            )
        )
    return out


def _apply_demand_delta(d: np.ndarray, start_min: int, end_min: int, delta: int) -> None:
    a = max(0, clock_to_step(start_min))
    b = min(N_STEPS, clock_to_step(end_min))
    d[a:b] = np.maximum(0, d[a:b] + delta)


def make_scenario(name: str, seed: int, mismatch: bool = False, inject: bool = False) -> Scenario:
    """Build a scenario deterministically from (name, seed)."""
    if name not in ALL_SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; choose from {ALL_SCENARIOS}")
    rng = _rng(name, seed)
    wording = "heldout" if seed >= 10 else "dev"
    corpus = _load_corpus()
    health = np.ones((N_STEPS, N_ROOMS))
    t_out = _base_tout(rng)
    events: list[dict[str, Any]] = []
    notices: list[Notice] = []
    unavailable: frozenset[str] = frozenset()

    if name == "steady":
        demand = _steady_demand(rng)
    elif name in ("surge", "heat", "fault", "fault_surge", "double_fault"):
        demand = _surge_demand(rng)
        if name != "surge":
            t_out, hs = _add_heat(t_out, rng)
            events.append({"time": clock_str(START_MIN + hs * STEP_MIN), "event": "outdoor temperature rises"})
        if name == "fault":
            _offline(health, "B201", 20 * 60)
            events.append({"time": "20:00", "event": "B201 AC offline"})
        elif name == "fault_surge":
            _offline(health, "B201", 19 * 60 + 30)
            events.append({"time": "19:30", "event": "B201 AC offline"})
        elif name == "double_fault":
            _offline(health, "B201", 20 * 60)
            _offline(health, "A102", 20 * 60)
            events.append({"time": "20:00", "event": "B201 and A102 AC offline"})
    elif name in ("notice_surge", "notice_fault"):
        demand = _steady_demand(rng)
        ev = 19 * 60 + 40 + 10 * int(rng.integers(-1, 2))
        if name == "notice_fault":
            ev = 19 * 60 + 40          # T1 timeline: N1 at 19:20, event 19:40, fault 20:00
        end = min(22 * 60, ev + 120)
        announced = 60
        realized = announced + int(rng.integers(-6, 7))
        _apply_demand_delta(demand, ev, end, realized)
        n1 = _demand_notice(rng, corpus, wording, "N1", ev, end, announced, up=True)
        if name == "notice_fault":
            n1.release_min = 19 * 60 + 20
            _offline(health, "B201", 20 * 60)
            events.append({"time": "20:00", "event": "B201 AC offline"})
            notices.append(
                Notice("R1", 20 * 60 + 10, _fmt(_pick(rng, corpus[wording]["report"]), room="B201"), kind="report")
            )
        notices.append(n1)
        notices += _distractors(rng, corpus, wording, 2, 1 + int(rng.integers(0, 2)))
        events.append({"time": clock_str(ev), "event": f"+{realized} students"})
    elif name == "notice_maintenance":
        unavailable = frozenset({"B201"})
        demand = np.full(N_STEPS, 150 + int(rng.integers(-5, 6)), dtype=int)
        ev = 20 * 60 + 10 * int(rng.integers(-1, 2))
        announced = 90
        realized = announced + int(rng.integers(-5, 6))
        _apply_demand_delta(demand, ev, 21 * 60 + 30, realized)
        demand = np.minimum(demand, 240)
        notices.append(_demand_notice(rng, corpus, wording, "N1", ev, 21 * 60 + 30, announced, up=True))
        notices += _distractors(rng, corpus, wording, 2, 1 + int(rng.integers(0, 2)))
        events.append({"time": "18:00", "event": "B201 under maintenance (230 seats available)"})
        events.append({"time": clock_str(ev), "event": f"+{realized} students"})
    elif name == "notice_drop":
        demand = _surge_demand(rng)
        demand[clock_to_step(21 * 60):] = demand[clock_to_step(21 * 60) - 1]   # no unannounced drop
        ev = 20 * 60 + 30 + 10 * int(rng.integers(-1, 2))
        announced = 50
        realized = announced + int(rng.integers(-5, 6))
        _apply_demand_delta(demand, ev, 22 * 60, -realized)
        notices.append(_demand_notice(rng, corpus, wording, "N1", ev, 22 * 60, announced, up=False))
        notices += _distractors(rng, corpus, wording, 2, 1 + int(rng.integers(0, 2)))
        events.append({"time": clock_str(ev), "event": f"-{realized} students"})
    else:  # pragma: no cover
        raise AssertionError(name)

    if inject:
        text = _pick(rng, corpus["injection"])
        rel = START_MIN + 10 * int(rng.integers(3, 15))
        notices.append(Notice("NX", rel, text, injected=True))

    mm = None
    if mismatch:
        mrng = _rng(name, seed, "mismatch")
        mm = {
            r: {k: float(1.0 + 0.10 * mrng.choice([-1.0, 1.0])) for k in ("R", "C", "COP")}
            for r in ROOM_IDS
        }
    notices.sort(key=lambda n: (n.release_min, n.notice_id))
    return Scenario(
        name=name,
        seed=seed,
        demand=demand.astype(int),
        t_out=t_out,
        health=health,
        unavailable=unavailable,
        notices=notices,
        mismatch=mm,
        events=events,
    )
