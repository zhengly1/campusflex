"""Student assignment ledger. It changes only after device acknowledgement (Section 3.5)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import N_ROOMS, PlannerConfig
from .model import AssignmentResult
from .validator import in_no_move_window


@dataclass
class Student:
    sid: int
    room: int | None = None        # None = present but waiting for a seat
    moves: int = 0
    last_move_step: int = -10**6


class Ledger:
    def __init__(self, rng: np.random.Generator, departure_mode: str = "random"):
        self.rng = rng
        self.departure_mode = departure_mode
        self.students: dict[int, Student] = {}
        self.next_id = 0
        self.departed = 0
        self.arrived = 0

    # ------------------------------------------------------------------ views
    def present(self) -> int:
        return len(self.students)

    def counts(self) -> np.ndarray:
        c = np.zeros(N_ROOMS, dtype=int)
        for s in self.students.values():
            if s.room is not None:
                c[s.room] += 1
        return c

    def movable(self, cfg: PlannerConfig, step: int, clock_min: int) -> np.ndarray:
        m = np.zeros(N_ROOMS, dtype=int)
        if in_no_move_window(clock_min, cfg):
            return m
        for s in self.students.values():
            if (
                s.room is not None
                and s.moves < cfg.max_moves_per_student
                and step - s.last_move_step >= cfg.min_student_dwell_steps
            ):
                m[s.room] += 1
        return m

    # ---------------------------------------------------------------- updates
    def set_demand(self, d: int) -> tuple[int, int]:
        """Arrivals join the waiting queue; departures leave (random or highest-id, v1)."""
        cur = self.present()
        if d > cur:
            for _ in range(d - cur):
                self.students[self.next_id] = Student(self.next_id)
                self.next_id += 1
            self.arrived += d - cur
            return d - cur, 0
        if d < cur:
            ids = sorted(self.students)
            k = cur - d
            if self.departure_mode == "highest_id":
                leave = ids[-k:]
            else:
                leave = list(self.rng.choice(ids, size=k, replace=False))
            for sid in leave:
                del self.students[int(sid)]
            self.departed += k
            return 0, k
        return 0, 0

    def _pick(self, room: int, k: int, step: int) -> list[Student]:
        cands = [s for s in self.students.values() if s.room == room]
        # Prefer students who have moved least and longest ago; stable by id.
        cands.sort(key=lambda s: (s.moves, s.last_move_step, s.sid))
        return cands[:k]

    def apply(self, res: AssignmentResult, step: int) -> None:
        """Map a count-level assignment onto individual students."""
        for f, t, k in res.pairs:
            for s in self._pick(f, k, step):
                s.room = t
                s.moves += 1
                s.last_move_step = step
        cur = self.counts()
        for i in range(N_ROOMS):
            excess = int(cur[i] - res.counts[i])
            if excess > 0:                      # loss of seat
                for s in sorted((s for s in self.students.values() if s.room == i), key=lambda s: -s.sid)[:excess]:
                    s.room = None
        waiting = sorted(sid for sid, s in self.students.items() if s.room is None)
        pos = 0
        for room, k in res.seated_new:
            for sid in waiting[pos : pos + k]:
                self.students[sid].room = room
            pos += k
        assert np.array_equal(self.counts(), res.counts), "ledger/assignment mismatch"
