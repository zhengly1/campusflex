"""Deterministic validator (Sections 3.1, 3.4, 3.6).

Every candidate – mask, rule-staged, LLM-staged or baseline – passes the same checks.
The validator rejects unknown rooms, capacity breaches, dwell violations and commands to
offline units. ``apply_plan`` re-runs ``validate_action`` and refuses anything that fails.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import ValidationError

from .config import ROOM_IDS, ROOM_INDEX, STEP_MIN, TRANSFER_C, PlannerConfig, parse_clock
from .plans import CompiledPlan, MacroPlan, StepAction, parse_macro
from .state import Snapshot

PRECOOL_MIN_C = 22.0


@dataclass(frozen=True)
class Violation:
    code: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "detail": self.detail}


def in_no_move_window(clock_min: int, cfg: PlannerConfig) -> bool:
    for a, b in cfg.no_move_windows:
        if parse_clock(a) <= clock_min < parse_clock(b):
            return True
    return False


def _unsafe(snap: Snapshot, i: int) -> bool:
    """Safety exemption: occupants of offline or over-threshold rooms may always be moved."""
    return snap.health[i] <= 0 or snap.temps[i] >= TRANSFER_C


def validate_macro(obj: Any, snap: Snapshot, cfg: PlannerConfig) -> tuple[MacroPlan | None, list[Violation]]:
    """Schema and static room checks on an (LLM-)proposed macro-plan, before compilation."""
    try:
        m = parse_macro(obj)
    except (ValidationError, ValueError, TypeError) as e:
        return None, [Violation("schema", str(e).splitlines()[0][:200])]
    v: list[Violation] = []
    horizon_min = cfg.rollout_horizon_min
    # Moves at one offset run in order (model.assign): "all" is whoever is still in the source
    # room after the earlier moves out of it, and moves into a room add up against its seats.
    left: dict[tuple[int, int], int] = {}
    into: dict[tuple[int, int], int] = {}
    for s in m.steps:
        off = s.offset_min
        if off % STEP_MIN != 0 or off >= horizon_min:
            v.append(Violation("bad_offset", f"offset {s.at} not a multiple of {STEP_MIN} below {horizon_min}"))
        named = list(s.open) + list(s.close)
        if s.precool:
            named.append(s.precool.room)
        if s.move:
            named += [s.move.from_, s.move.to]
        for r in named:
            if r not in ROOM_INDEX:
                v.append(Violation("unknown_room", f"unknown room {r!r}"))
        for r in list(s.open) + ([s.precool.room] if s.precool else []) + ([s.move.to] if s.move else []):
            if r in ROOM_INDEX:
                i = ROOM_INDEX[r]
                if snap.health[i] <= 0:
                    v.append(Violation("offline_room", f"{r} AC is offline"))
                if snap.capacity[i] <= 0:
                    v.append(Violation("unavailable_room", f"{r} is unavailable"))
        if s.precool and s.precool.room in ROOM_INDEX:
            t = s.precool.target_c
            if not (PRECOOL_MIN_C <= t <= cfg.band_high_c):
                v.append(Violation("precool_range", f"precool target {t} outside [{PRECOOL_MIN_C}, {cfg.band_high_c}]"))
        if s.move and s.move.from_ in ROOM_INDEX and s.move.to in ROOM_INDEX:
            f, t = ROOM_INDEX[s.move.from_], ROOM_INDEX[s.move.to]
            c = s.move.count
            rest = left.setdefault((off, f), int(snap.counts[f]))
            need = rest if c == "all" else int(c)
            if isinstance(c, int) and c <= 0:
                v.append(Violation("bad_count", f"move count {c} must be positive"))
            got = into.get((off, t), 0)
            if f != t and got + need > int(snap.capacity[t]):
                v.append(Violation("capacity", f"move of {need} into {s.move.to} exceeds {int(snap.capacity[t])} seats"
                                               + (f" ({got} already moved in at {s.at})" if got else "")))
            if f == t:
                v.append(Violation("bad_move", "move source equals target"))
            else:
                left[(off, f)] = max(0, rest - max(0, need))
                into[(off, t)] = got + max(0, need)
    # dedupe while preserving order
    seen, out = set(), []
    for x in v:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return m, out


def validate_action(a: StepAction, snap: Snapshot, cfg: PlannerConfig) -> list[Violation]:
    """Checks on the action that would be committed now, against the current snapshot."""
    v: list[Violation] = []
    rooms = set(a.open) | {r for r, _ in a.setpoints} | {f for f, _, _ in a.moves} | {t for _, t, _ in a.moves}
    unknown = sorted(r for r in rooms if r not in ROOM_INDEX)
    if unknown:
        return [Violation("unknown_room", f"unknown room(s) {unknown}")]
    new_open = a.open_mask()
    # Commands to offline units (Eq. 1: 0 ≤ u ≤ h)
    for r, sp in a.setpoints:
        if snap.health[ROOM_INDEX[r]] <= 0:
            v.append(Violation("offline_command", f"AC command to offline unit {r}"))
        if not (PRECOOL_MIN_C <= float(sp) <= cfg.band_high_c):
            v.append(Violation("setpoint_range", f"setpoint {sp} for {r} outside [{PRECOOL_MIN_C}, {cfg.band_high_c}]"))
    stale = getattr(snap, "stale", None)
    for r in a.open:
        i = ROOM_INDEX[r]
        if snap.capacity[i] <= 0:
            v.append(Violation("unavailable_room", f"{r} is unavailable"))
        if stale is not None and bool(stale[i]) and not snap.open[i]:
            v.append(Violation("stale_room", f"{r} has no fresh sensor reading; not opened"))
    # Room dwell
    for i, r in enumerate(ROOM_IDS):
        if bool(new_open[i]) != bool(snap.open[i]):
            closing_unsafe = (not new_open[i]) and _unsafe(snap, i)
            if snap.steps_since_switch[i] < cfg.min_room_dwell_steps and not closing_unsafe:
                v.append(Violation("dwell_room", f"{r} switched {int(snap.steps_since_switch[i])} step(s) ago"))
    # Student dwell / max moves / no-move windows
    out_needed = np.zeros(len(ROOM_IDS), dtype=int)
    for i in range(len(ROOM_IDS)):
        if snap.counts[i] > 0 and not new_open[i] and not _unsafe(snap, i):
            out_needed[i] += int(snap.counts[i])
    free = np.where(new_open, snap.capacity - snap.counts, 0)
    left = snap.counts.astype(int).copy()      # "all" = whoever is still there after earlier moves (model.assign)
    for f, t, c in a.moves:
        fi, ti = ROOM_INDEX[f], ROOM_INDEX[t]
        if fi == ti:
            v.append(Violation("bad_move", "move source equals target"))
            continue
        if not new_open[ti]:
            v.append(Violation("move_to_closed", f"move into closed room {t}"))
        k = int(left[fi]) if c == "all" else int(c)
        if c != "all" and k <= 0:
            v.append(Violation("bad_count", f"move count {c} must be positive"))
        if k > int(free[ti]):
            v.append(Violation("capacity", f"move of {k} into {t} exceeds {int(free[ti])} free seats"))
        if c != "all" and k > int(left[fi]):
            v.append(Violation("capacity", f"move of {k} from {f} exceeds {int(left[fi])} occupants"))
        moved = max(0, min(k, int(left[fi])))
        left[fi] -= moved
        # A closing room already counts all its occupants; do not count its move-outs twice.
        if not _unsafe(snap, fi) and new_open[fi]:
            out_needed[fi] += min(k, int(snap.counts[fi]))
        free[ti] -= k
        if new_open[fi]:
            free[fi] += moved                   # seats freed in the source (model.assign runs moves in order)
    window = in_no_move_window(snap.clock_min, cfg)
    for i, r in enumerate(ROOM_IDS):
        if out_needed[i] <= 0:
            continue
        if window:
            v.append(Violation("no_move_window", f"moving {int(out_needed[i])} from {r} inside a no-move window"))
        elif out_needed[i] > snap.movable[i]:
            v.append(
                Violation(
                    "dwell_student",
                    f"{int(out_needed[i])} moves from {r} but only {int(snap.movable[i])} students are movable",
                )
            )
    return v


def later_step_violations(a: StepAction, open_prev: np.ndarray, counts: np.ndarray, capacity: np.ndarray,
                          temps: np.ndarray, health: np.ndarray, clock_min: int, cfg: PlannerConfig,
                          at: str) -> list[Violation]:
    """The occupancy checks of :func:`validate_action` for a later plan step, against the state the
    rollout predicts at that step (``counts`` before the step's moves).

    ``validate_action`` will run these checks on the real snapshot when the step comes due; a plan
    whose later step already fails them on its own prediction is scored on moves the validator
    would refuse (the rollout trims a move to the free seats), so it is rejected instead.
    """
    v: list[Violation] = []
    mask = a.open_mask()
    unsafe = (health <= 0) | (temps >= TRANSFER_C)
    free = np.where(mask, capacity - counts, 0).astype(int)
    left = counts.astype(int).copy()
    moved_out = np.zeros(len(ROOM_IDS), dtype=int)
    for f, t, c in a.moves:
        if f not in ROOM_INDEX or t not in ROOM_INDEX or f == t:
            continue                                   # reported by validate_macro / validate_plan
        fi, ti = ROOM_INDEX[f], ROOM_INDEX[t]
        if not mask[ti]:
            v.append(Violation("move_to_closed", f"move into closed room {t} at {at}"))
        k = int(left[fi]) if c == "all" else int(c)
        if k > int(free[ti]):
            v.append(Violation("capacity", f"move of {k} into {t} at {at} exceeds the {int(free[ti])} free "
                                           f"seats predicted then"))
        moved = max(0, min(k, int(left[fi])))
        left[fi] -= moved
        moved_out[fi] += moved
        free[ti] -= k
        if mask[fi]:
            free[fi] += moved
    if in_no_move_window(clock_min, cfg):
        closing = [ROOM_IDS[i] for i in range(len(ROOM_IDS))
                   if open_prev[i] and not mask[i] and counts[i] - moved_out[i] > 0 and not unsafe[i]]
        if closing:
            v.append(Violation("no_move_window", f"closing occupied {', '.join(closing)} at {at} inside a "
                                                 f"no-move window"))
    return v


def validate_plan(plan: CompiledPlan, snap: Snapshot, cfg: PlannerConfig) -> list[Violation]:
    """Full plan check before simulation.

    The first action is checked against the snapshot. Later actions are checked for static
    rules (offline units, unavailable rooms, setpoint range) and for the dwell rules over the
    plan itself: a room may not switch again within ``min_room_dwell_steps`` of its last
    switch, and students moved into a room may not be moved out (or displaced by closing it)
    within ``min_student_dwell_steps``. Moves inside a no-move window are rejected.
    """
    if not plan.actions:
        return [Violation("empty_plan", "plan has no actions")]
    v = validate_action(plan.first(), snap, cfg)
    n = len(ROOM_IDS)
    since = snap.steps_since_switch.astype(int).copy()
    prev_open = plan.actions[0].open_mask()
    since = np.where(prev_open != snap.open.astype(bool), 0, since) + 1
    last_in = np.full(n, -10**6, dtype=int)             # step of the last move into each room
    for f, t, c in plan.actions[0].moves:
        if t in ROOM_INDEX:
            last_in[ROOM_INDEX[t]] = 0
    for k, a in enumerate(plan.actions[1:], start=1):
        at = f"+{k * STEP_MIN}"
        for r, sp in a.setpoints:
            if r not in ROOM_INDEX:
                v.append(Violation("unknown_room", f"unknown room {r!r} at {at}"))
                continue
            if snap.health[ROOM_INDEX[r]] <= 0:
                v.append(Violation("offline_command", f"AC command to offline unit {r} at {at}"))
            if not (PRECOOL_MIN_C <= float(sp) <= cfg.band_high_c):
                v.append(Violation("setpoint_range", f"setpoint {sp} for {r} at {at}"))
        for r in a.open:
            if r in ROOM_INDEX and snap.capacity[ROOM_INDEX[r]] <= 0:
                v.append(Violation("unavailable_room", f"{r} unavailable at {at}"))
        mask = a.open_mask()
        switched = mask != prev_open
        for i in np.flatnonzero(switched):
            exempt = (not mask[i]) and _unsafe(snap, i)
            if since[i] < cfg.min_room_dwell_steps and not exempt:
                v.append(Violation("dwell_room", f"{ROOM_IDS[i]} switched again at {at} ({int(since[i])} step(s) after)"))
            if not mask[i] and not _unsafe(snap, i) and k - last_in[i] < cfg.min_student_dwell_steps:
                v.append(Violation("dwell_student", f"closing {ROOM_IDS[i]} at {at} displaces students moved in "
                                                    f"{k - int(last_in[i])} step(s) earlier"))
        if a.moves and in_no_move_window(snap.clock_min + k * STEP_MIN, cfg):
            v.append(Violation("no_move_window", f"moves at {at} inside a no-move window"))
        for f, t, c in a.moves:
            if f not in ROOM_INDEX or t not in ROOM_INDEX:
                v.append(Violation("unknown_room", f"unknown room in move at {at}"))
                continue
            fi = ROOM_INDEX[f]
            if not mask[ROOM_INDEX[t]]:
                v.append(Violation("move_to_closed", f"move into closed room {t} at {at}"))
            if not _unsafe(snap, fi) and k - last_in[fi] < cfg.min_student_dwell_steps:
                v.append(Violation("dwell_student", f"moving out of {f} at {at}, {k - int(last_in[fi])} step(s) "
                                                    f"after students were moved in"))
        for f, t, c in a.moves:
            if t in ROOM_INDEX:
                last_in[ROOM_INDEX[t]] = k
        since = np.where(switched, 0, since) + 1
        prev_open = mask
    seen, out = set(), []
    for x in v:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
