"""Usage profiles per room and time slot, learned from how people adjust the devices (AI).

What is remembered, per (room, hourly slot): the preferred AC setpoint and fan speed, the
preferred desk illuminance, the curtain position and how often the projector is used.

Learning
  * an explicit manual adjustment is strong evidence (weight 1) for the value people chose;
  * a setting left untouched for a whole occupied slot is weak evidence (weight
    ``implicit_weight``) that it was acceptable;
  * evidence fades by ``decay`` per day, so changed habits take over;
  * a room's estimate is shrunk toward the campus-wide estimate of the same slot
    (partial pooling with ``prior_strength`` pseudo-observations), so a room that was
    rarely used still gets the slot's habits; with no data at all the system defaults apply.

Profiles persist as JSON (``save``/``load``): the scheduler starts every evening from what
earlier evenings taught it. ``personas`` labels each slot from the learned values and
``narrate`` writes them as text (template, or an LLM whose numbers are checked by the
faithfulness checker).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..config import END_MIN, START_MIN, clock_str

SLOT_MIN = 60
SLOTS = tuple((a, a + SLOT_MIN) for a in range(START_MIN, END_MIN, SLOT_MIN))
ATTRS = ("setpoint", "fan", "lux", "curtain", "projector")
DEFAULTS = {"setpoint": 25.0, "fan": 2.0, "lux": 400.0, "curtain": 30.0, "projector": 0.2}
DISPLAY_DIGITS = {"setpoint": 1, "fan": 1, "lux": 0, "curtain": 0, "projector": 2}


def slot_of(minute: int) -> int:
    return max(0, min(len(SLOTS) - 1, (int(minute) - START_MIN) // SLOT_MIN))


def slot_label(k: int) -> str:
    a, b = SLOTS[k]
    return f"{clock_str(a)}-{clock_str(b)}"


@dataclass
class Belief:
    """Weighted running mean and variance with exponential forgetting."""

    w: float = 0.0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, x: float, weight: float = 1.0) -> None:
        if weight <= 0:
            return
        w = self.w + weight
        d = float(x) - self.mean
        self.mean += weight / w * d
        self.m2 += weight * d * (float(x) - self.mean)
        self.w = w

    def decay(self, f: float) -> None:
        self.w *= f
        self.m2 *= f

    @property
    def sd(self) -> float | None:
        return (self.m2 / self.w) ** 0.5 if self.w > 1e-9 else None


class ProfileStore:
    VERSION = 1

    def __init__(self, decay: float = 0.97, prior_strength: float = 2.0, implicit_weight: float = 0.2):
        self.decay, self.prior_strength, self.implicit_weight = decay, prior_strength, implicit_weight
        self.cells: dict[str, Belief] = {}
        self.pool: dict[str, Belief] = {}
        self.days = 0
        self.n_adjustments = 0

    # ---------------------------------------------------------------- learning
    def observe(self, room: str, slot: int, attr: str, value: float, weight: float = 1.0) -> None:
        if attr not in ATTRS:
            raise ValueError(f"unknown attribute {attr!r}")
        self.cells.setdefault(f"{room}|{slot}|{attr}", Belief()).add(value, weight)
        self.pool.setdefault(f"{slot}|{attr}", Belief()).add(value, weight)

    def observe_adjustment(self, room: str, slot: int, device: str, value: dict[str, Any],
                           lux_after: float | None = None) -> None:
        """A manual adjustment: what people chose is what they want in this room and slot."""
        if device == "ac" and value.get("setpoint") is not None:
            self.observe(room, slot, "setpoint", float(value["setpoint"]))
            self.observe(room, slot, "fan", float(value["fan"]))
        elif device == "light" and lux_after is not None:
            self.observe(room, slot, "lux", float(lux_after))
        elif device == "curtain":
            self.observe(room, slot, "curtain", float(value["open_pct"]))
        elif device == "projector":
            self.observe(room, slot, "projector", 1.0 if value["on"] else 0.0)
        else:
            return
        self.n_adjustments += 1

    def observe_accepted(self, room: str, slot: int, settings: dict[str, float]) -> None:
        """Settings that stayed untouched through an occupied slot (weak evidence)."""
        for attr, v in settings.items():
            self.observe(room, slot, attr, float(v), self.implicit_weight)

    def end_day(self) -> None:
        for b in list(self.cells.values()) + list(self.pool.values()):
            b.decay(self.decay)
        self.days += 1

    # --------------------------------------------------------------- recall
    def estimate(self, room: str, slot: int, attr: str) -> tuple[float, float, str]:
        """(value, evidence weight of this room, source: room | slot | default)."""
        pool = self.pool.get(f"{slot}|{attr}")
        slot_mean = pool.mean if pool is not None and pool.w > 0.5 else DEFAULTS[attr]
        cell = self.cells.get(f"{room}|{slot}|{attr}")
        if cell is not None and cell.w > 1e-6:
            k = self.prior_strength
            return (cell.w * cell.mean + k * slot_mean) / (cell.w + k), cell.w, "room"
        if pool is not None and pool.w > 0.5:
            return slot_mean, 0.0, "slot"
        return DEFAULTS[attr], 0.0, "default"

    def recommend(self, room: str, slot: int) -> dict[str, Any]:
        """Device settings to start from in ``room`` during ``slot``."""
        e = {a: self.estimate(room, slot, a) for a in ATTRS}
        return {
            "setpoint": round(min(27.0, max(22.0, e["setpoint"][0])), 1),
            "fan": int(min(3, max(1, round(e["fan"][0])))),
            "lux": float(e["lux"][0]),
            "curtain": int(min(100, max(0, round(e["curtain"][0] / 10.0) * 10))),
            "projector": e["projector"][0] >= 0.5,
            "projector_p": float(e["projector"][0]),
            "evidence": {a: round(e[a][1], 2) for a in ATTRS},
            "source": {a: e[a][2] for a in ATTRS},
        }

    # ------------------------------------------------------------- personas
    def personas(self, rooms: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        """One persona per slot (campus-wide), with the rooms that deviate from it."""
        out = []
        for k in range(len(SLOTS)):
            vals = {a: (self.pool[f"{k}|{a}"].mean if f"{k}|{a}" in self.pool and self.pool[f"{k}|{a}"].w > 0.5
                        else None) for a in ATTRS}
            n = round(sum(self.pool[f"{k}|{a}"].w for a in ATTRS if f"{k}|{a}" in self.pool), 1)
            room_dev = []
            for r in rooms or sorted({key.split("|")[0] for key in self.cells}):
                rec = self.recommend(r, k)
                if rec["source"]["setpoint"] != "room":
                    continue
                d = {}
                if vals["setpoint"] is not None and abs(rec["setpoint"] - vals["setpoint"]) >= 0.3:
                    d["setpoint"] = rec["setpoint"]
                if vals["lux"] is not None and abs(rec["lux"] - vals["lux"]) >= 40:
                    d["lux"] = round(rec["lux"])
                if vals["fan"] is not None and rec["fan"] != round(vals["fan"]):
                    d["fan"] = rec["fan"]
                if d:
                    room_dev.append({"room": r, **d})
            out.append({"slot": slot_label(k), "evidence": n,
                        "values": {a: _display(a, v) for a, v in vals.items()},
                        "tags": persona_tags(vals), "rooms_that_differ": room_dev})
        return out

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        return {"version": self.VERSION, "slots": [slot_label(k) for k in range(len(SLOTS))],
                "decay": self.decay, "prior_strength": self.prior_strength, "implicit_weight": self.implicit_weight,
                "days": self.days, "n_adjustments": self.n_adjustments,
                "cells": {k: asdict(v) for k, v in sorted(self.cells.items())},
                "pool": {k: asdict(v) for k, v in sorted(self.pool.items())}}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProfileStore":
        if d.get("version") != cls.VERSION:
            raise ValueError(f"profile store version {d.get('version')} is not {cls.VERSION}")
        s = cls(d["decay"], d["prior_strength"], d["implicit_weight"])
        s.days, s.n_adjustments = int(d["days"]), int(d["n_adjustments"])
        s.cells = {k: Belief(**v) for k, v in d["cells"].items()}
        s.pool = {k: Belief(**v) for k, v in d["pool"].items()}
        return s

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "ProfileStore":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _display(attr: str, v: float | None) -> float | int | None:
    if v is None:
        return None
    d = DISPLAY_DIGITS[attr]
    return int(round(v)) if d == 0 else round(v, d)


def persona_tags(v: dict[str, float | None]) -> list[str]:
    tags = []
    if v.get("projector") is not None:
        tags.append("lecture (projector on)" if v["projector"] >= 0.6 else
                    "no projector" if v["projector"] <= 0.3 else "projector sometimes")
    if v.get("lux") is not None:
        tags.append("bright for reading" if v["lux"] >= 460 else "dim for the screen" if v["lux"] <= 330 else
                    "medium light")
    if v.get("setpoint") is not None:
        tags.append("prefers cool" if v["setpoint"] <= 24.7 else "tolerates warm" if v["setpoint"] >= 25.4 else
                    "moderate temperature")
    if v.get("fan") is not None:
        tags.append("strong airflow" if v["fan"] >= 2.4 else "quiet fan" if v["fan"] <= 1.4 else "medium fan")
    if v.get("curtain") is not None:
        tags.append("curtains open for daylight" if v["curtain"] >= 40 else "curtains closed" if v["curtain"] <= 10
                    else "curtains half open")
    return tags


# ------------------------------------------------------------------ narration
NARRATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"lines": {"type": "array", "items": {"type": "string"}}},
    "required": ["lines"],
    "additionalProperties": False,
}
NARRATE_SYSTEM = """You describe how people use the study rooms of a campus in each evening time slot, \
for the building administrator. You receive the learned usage profiles as JSON. Write one plain \
sentence per slot, in the same order. Use only numbers that appear in the JSON, written as they \
appear; do not compute new numbers. Every number is checked automatically and unmatched numbers \
are removed."""


def template_lines(personas: list[dict[str, Any]]) -> list[str]:
    out = []
    for p in personas:
        v = p["values"]
        if v["setpoint"] is None:
            out.append(f"{p['slot']}: no usage learned yet.")
            continue
        out.append(f"{p['slot']} ({', '.join(p['tags'])}): usually {v['setpoint']} °C, fan {v['fan']}, "
                   f"{v['lux']} lux, curtains {v['curtain']} % open, projector share {v['projector']}"
                   + ("; rooms that differ: " + ", ".join(
                       f"{d['room']} " + " ".join(f"{k} {val}" for k, val in d.items() if k != "room")
                       for d in p["rooms_that_differ"]) if p["rooms_that_differ"] else "") + ".")
    return out


def narrate(store: ProfileStore, client: Any = None) -> dict[str, Any]:
    """Text descriptions of the slot personas: LLM when ``client`` is given (numbers checked),
    otherwise the template."""
    personas = store.personas()
    tmpl = template_lines(personas)
    if client is None:
        return {"source": "template", "lines": tmpl, "personas": personas}
    from ..llm.client import CallBudget, call
    from ..roles.faithfulness import check

    r = call(client, CallBudget(1, 60.0), role="L4", system=NARRATE_SYSTEM,
             user="Learned profiles (tool output):\n" + json.dumps(personas, indent=1), schema=NARRATE_SCHEMA)
    if not r.ok or not isinstance(r.data, dict) or not isinstance(r.data.get("lines"), list):
        return {"source": "template", "lines": tmpl, "personas": personas, "llm_error": r.error}
    lines, flagged = [], []
    for t in r.data["lines"]:
        fr = check(str(t), personas)
        lines.append(fr.cleaned_text)
        flagged += fr.flagged
    return {"source": "llm", "model": r.model, "lines": lines, "flagged_numbers": flagged, "personas": personas}
