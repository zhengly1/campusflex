"""Actions, the staged macro-plan language and its deterministic compiler (Section 3.4, L2).

A macro-plan is a short list of timed steps (open, close, precool, move). The compiler
expands it into one ``StepAction`` per rollout step: room mask, AC setpoints and moves.
Single-mask candidates are the special case of one action held for the whole horizon.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import N_ROOMS, ROOM_IDS, ROOM_INDEX, STEP_MIN, PlannerConfig


@dataclass(frozen=True)
class StepAction:
    """What is committed for one ten-minute interval."""

    open: frozenset[str]
    setpoints: tuple[tuple[str, float], ...] = ()      # rooms whose AC runs, with target °C
    moves: tuple[tuple[str, str, Union[int, str]], ...] = ()   # (from, to, count | "all")
    fill: str = "smart"          # seat-assignment rule: "smart" | "inventory" | "efficiency"

    def open_mask(self) -> np.ndarray:
        return np.array([r in self.open for r in ROOM_IDS], dtype=bool)

    def setpoint_array(self) -> np.ndarray:
        sp = np.full(N_ROOMS, np.nan)
        for r, v in self.setpoints:
            sp[ROOM_INDEX[r]] = v
        return sp

    def move_indices(self) -> list[tuple[int, int, int]]:
        return [(ROOM_INDEX[f], ROOM_INDEX[t], -1 if c == "all" else int(c)) for f, t, c in self.moves]

    def to_dict(self) -> dict[str, Any]:
        return {
            "open": sorted(self.open),
            "setpoints": {r: v for r, v in self.setpoints},
            "moves": [{"from": f, "to": t, "count": c} for f, t, c in self.moves],
            "fill": self.fill,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StepAction":
        return cls(
            open=frozenset(d.get("open", [])),
            setpoints=tuple(sorted((str(k), float(v)) for k, v in d.get("setpoints", {}).items())),
            moves=tuple((m["from"], m["to"], m["count"]) for m in d.get("moves", [])),
            fill=d.get("fill", "smart"),
        )


@dataclass
class CompiledPlan:
    plan_id: str
    source: str                         # "mask" | "rule" | "L2" | "carry:<src>" | baseline name
    actions: list[StepAction]
    macro: dict[str, Any] | None = None
    rationale: str = ""                 # logged, never used in scoring
    origin_step: int = 0

    def first(self) -> StepAction:
        return self.actions[0]

    def action_at(self, k: int) -> StepAction:
        """Action for rollout step k; the final action is held after the plan ends (without moves)."""
        if k < len(self.actions):
            return self.actions[k]
        last = self.actions[-1]
        return StepAction(open=last.open, setpoints=last.setpoints, moves=(), fill=last.fill)

    def shifted(self, k: int = 1) -> "CompiledPlan | None":
        """Carry-forward: the remainder of a multi-step plan at the next decision step."""
        if len(self.actions) <= k:
            return None
        rest = self.actions[k:]
        if all(a == rest[0] and not a.moves for a in rest):
            # Nothing staged remains: a plain mask, already in the candidate pool.
            return None
        base = self.source.split(":", 1)[-1]
        return CompiledPlan(
            plan_id=f"{self.plan_id}>{k}",
            source=f"carry:{base}",
            actions=list(rest),
            macro=self.macro,
            rationale=self.rationale,
            origin_step=self.origin_step,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "source": self.source,
            "first": self.first().to_dict(),
            "n_actions": len(self.actions),
        }


# ---------------------------------------------------------------------------
# Macro-plan language (JSON) – the schema the L2 proposer must emit.
# ---------------------------------------------------------------------------
class Precool(BaseModel):
    model_config = ConfigDict(extra="forbid")
    room: str
    target_c: float


class Move(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    from_: str = Field(alias="from")
    to: str
    count: Union[int, Literal["all"]]


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    at: str
    open: list[str] = Field(default_factory=list)
    close: list[str] = Field(default_factory=list)
    precool: Precool | None = None
    move: Move | None = None

    @field_validator("at")
    @classmethod
    def _offset(cls, v: str) -> str:
        if not re.fullmatch(r"\+\d{1,3}", v.strip()):
            raise ValueError("offset must look like '+0', '+20'")
        return v.strip()

    @property
    def offset_min(self) -> int:
        return int(self.at[1:])


class MacroPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plan_id: str
    steps: list[PlanStep] = Field(min_length=1, max_length=8)
    rationale: str = ""


MACRO_PLAN_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "plan_id": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "at": {"type": "string", "description": "offset in minutes, e.g. '+0', '+20'"},
                    "open": {"type": "array", "items": {"type": "string"}},
                    "close": {"type": "array", "items": {"type": "string"}},
                    "precool": {
                        "anyOf": [
                            {"type": "null"},
                            {
                                "type": "object",
                                "properties": {"room": {"type": "string"}, "target_c": {"type": "number"}},
                                "required": ["room", "target_c"],
                                "additionalProperties": False,
                            },
                        ]
                    },
                    "move": {
                        "anyOf": [
                            {"type": "null"},
                            {
                                "type": "object",
                                "properties": {
                                    "from": {"type": "string"},
                                    "to": {"type": "string"},
                                    "count": {"anyOf": [{"type": "integer"}, {"type": "string", "enum": ["all"]}]},
                                },
                                "required": ["from", "to", "count"],
                                "additionalProperties": False,
                            },
                        ]
                    },
                },
                "required": ["at", "open", "close", "precool", "move"],
                "additionalProperties": False,
            },
        },
        "rationale": {"type": "string"},
    },
    "required": ["plan_id", "steps", "rationale"],
    "additionalProperties": False,
}


def parse_macro(obj: Any) -> MacroPlan:
    """Parse a macro-plan from a dict or JSON string (raises pydantic.ValidationError / ValueError)."""
    if isinstance(obj, MacroPlan):
        return obj
    if isinstance(obj, (str, bytes)):
        obj = json.loads(obj)
    return MacroPlan.model_validate(obj)


def macro_to_dict(m: MacroPlan) -> dict[str, Any]:
    return m.model_dump(by_alias=True, exclude_none=True)


def compile_macro(
    macro: MacroPlan,
    open_now: frozenset[str],
    health: np.ndarray,
    cfg: PlannerConfig,
    horizon_steps: int,
    source: str = "L2",
    origin_step: int = 0,
) -> CompiledPlan:
    """Expand a macro-plan into per-step masks, setpoints and moves.

    Rooms must already be known (the validator runs before compilation). AC setpoints are
    issued only for units with health > 0; open rooms default to the comfort target and a
    precool target persists for the rest of the plan.
    """
    by_step: dict[int, list[PlanStep]] = {}
    for s in macro.steps:
        by_step.setdefault(s.offset_min // STEP_MIN, []).append(s)
    last_k = max(by_step) if by_step else 0
    n_actions = max(horizon_steps, last_k + 1)
    is_open = set(open_now)
    precool: dict[str, float] = {}
    actions: list[StepAction] = []
    for k in range(n_actions):
        moves: list[tuple[str, str, Union[int, str]]] = []
        for s in by_step.get(k, []):
            is_open |= set(s.open)
            for r in s.close:
                is_open.discard(r)
                precool.pop(r, None)
            if s.precool is not None:
                precool[s.precool.room] = float(s.precool.target_c)
            if s.move is not None:
                moves.append((s.move.from_, s.move.to, s.move.count))
        sp = []
        for r in ROOM_IDS:
            if health[ROOM_INDEX[r]] <= 0:
                continue
            if r in precool:
                sp.append((r, precool[r]))
            elif r in is_open:
                sp.append((r, cfg.target_c))
        actions.append(StepAction(open=frozenset(is_open), setpoints=tuple(sp), moves=tuple(moves)))
    return CompiledPlan(
        plan_id=macro.plan_id,
        source=source,
        actions=actions,
        macro=macro_to_dict(macro),
        rationale=macro.rationale,
        origin_step=origin_step,
    )


def mask_action(mask: frozenset[str], health: np.ndarray, cfg: PlannerConfig) -> StepAction:
    sp = tuple((r, cfg.target_c) for r in ROOM_IDS if r in mask and health[ROOM_INDEX[r]] > 0)
    return StepAction(open=frozenset(mask), setpoints=sp)


def mask_plan(mask: frozenset[str], health: np.ndarray, cfg: PlannerConfig, horizon_steps: int, step: int = 0) -> CompiledPlan:
    a = mask_action(mask, health, cfg)
    pid = "m:" + ("".join(r for r in ROOM_IDS if r in mask) or "none")
    return CompiledPlan(plan_id=pid, source="mask", actions=[a] * horizon_steps, origin_step=step)
