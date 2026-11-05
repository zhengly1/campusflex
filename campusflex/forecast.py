"""Demand and weather forecasts (Section 3.2 mechanism 2, role L3).

* ``ConstantForecaster`` – v1: hold current demand and weather.
* ``NoticeForecaster`` – rule forecast plus bounded, confidence-weighted adjustments
  produced by an interpreter (LLM L3, keyword parser or oracle) from released notices.

Adjustments shift demand relative to *now*: Δ(τ) = Σ w·δ·(active(τ) − active(now)), so an
event that has already started is not double counted. Each delta and the total are
clipped to ±cap (80 students); the planner never applies a raw delta.

Deterministic guards applied to every interpreter's output (``NoticeForecaster.update``):

* grounding (``ground_adjustment``): the adjustment cites a released notice that does not
  read like an instruction to the system, quotes its evidence verbatim, takes |delta| from a
  *count phrase* of that notice ("about 60 students", "around 40 attendees"; a temperature
  phrase for weather) – never from a clock time or any other number – and anchors its start
  to a whole clock token of the notice;
* at most one adjustment per (source_id, kind);
* contradictions (Table 10, "Two notices contradict each other → use the later notice; L4
  flags it to the administrator"): two adjustments of the same kind whose windows overlap and
  that concern the same place are two statements about the same event, so only the later
  notice is kept and the pair is reported in ``Forecast.conflicts``; L4 turns each entry into
  an administrator alert (``l4_negotiator.conflict_alerts``). Notices that name different
  places (LT-A vs Hall C, B201 vs A102) are independent events and both apply. A notice that
  names no known place is treated as being about the same event (the PDF's rule).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .config import ROOM_IDS, STEP_MIN, PlannerConfig, parse_clock
from .scenarios import Notice
from .state import Snapshot

CONFIDENCE_WEIGHT = {"high": 1.0, "medium": 0.7, "low": 0.4}


@dataclass
class Forecast:
    demand: np.ndarray                 # (L,)  index 0 = current step
    t_out: np.ndarray                  # (L+1,)
    adjustments: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)


class Forecaster(Protocol):
    def forecast(self, snap: Snapshot, length: int) -> Forecast: ...


class ConstantForecaster:
    name = "constant"

    def forecast(self, snap: Snapshot, length: int) -> Forecast:
        return Forecast(
            demand=np.full(length, int(snap.demand), dtype=int),
            t_out=np.full(length + 1, float(snap.t_out)),
        )


class Interpreter(Protocol):
    """Turns released notices into structured adjustments (one call per new notice batch)."""

    name: str

    def interpret(self, notices: list[Notice], snap: Snapshot) -> list[dict[str, Any]]: ...


# ------------------------------------------------------------------ text screens / grounding
# Deterministic injection screen (defence in depth on top of the prompt). Applied before the
# LLM call (flagged notices are not sent) and again when grounding any interpreter's output.
_INSTRUCTION = re.compile(
    r"\b(?:ignore|disregard|override|overrides|overriding|system\s+message|system\s+prompt|your\s+instructions|"
    r"new\s+instructions|previous\s+(?:rules|instructions|notices)|prior\s+(?:rules|instructions|notices)|"
    r"set\s+(?:the\s+)?delta|report\s+(?:a\s+|the\s+)?(?:delta|demand|kind)|output\s+(?:a\s+)?plan|"
    r"assistant|you\s+must|do\s+not\s+follow|(?:delta|kind|confidence|source_id)\s*[=:])"
    r"|(?:^|[\s(\[\"'])(?:system|admin|administrator|developer|assistant|ai)\s*:",
    re.I,
)

# A clock token as written: 19:40, 7:40, 7:40 pm, 7 pm (not part of a longer number).
_CLOCK_TOKEN = re.compile(
    r"(?<![\d:.])(\d{1,2})(?::([0-5]\d))?(?![\d:])(?:\s*([ap])\.?\s?m\b\.?)?",
    re.I,
)
# Count phrases: "about 60", "around 40", "roughly 30", "~25", "60 students", "40 attendees".
_COUNT_PHRASE = re.compile(
    r"(?:\b(?:about|around|roughly|approximately|approx\.?|estimate[sd]?|nearly|almost|some|up\s+to|"
    r"at\s+least|over)\s+|~\s*)(\d{1,4})(?!\d|[:.]\d)"
    r"|(?<![\d:.])\b(\d{1,4})(?!\d|[:.]\d)\s+(?:more\s+|additional\s+|extra\s+|other\s+)?"
    r"(?:students|attendees|members|riders|people|participants|occupants|visitors|persons)\b",
    re.I,
)
# Temperature phrases for weather adjustments: "3 °C", "2.5 degrees", "4 C".
_TEMP_PHRASE = re.compile(r"(?<![\d:.])(\d{1,2}(?:\.\d+)?)\s*(?:°\s*C?|degrees?\b|deg\b|C\b)", re.I)
_CLOCK_STRICT = re.compile(r"^\s*([01]?\d|2[0-3]):([0-5]\d)\s*$")


def looks_like_instruction(text: str) -> bool:
    return bool(_INSTRUCTION.search(str(text)))


def clock_tokens(text: str) -> set[int]:
    """Minutes of day of every clock time written in ``text`` (HH:MM, or H[:MM] am/pm)."""
    out: set[int] = set()
    for m in _CLOCK_TOKEN.finditer(str(text)):
        h, mm, ampm = int(m.group(1)), m.group(2), m.group(3)
        if mm is None and ampm is None:
            continue                                   # a bare number is not a clock time
        if ampm:
            if not 1 <= h <= 12:
                continue
            h = (h % 12) + (12 if ampm.lower() == "p" else 0)
        if h > 23:
            continue
        out.add(h * 60 + int(mm or 0))
    return out


def mask_clocks(text: str) -> str:
    """``text`` with every clock token blanked out (so its digits never count as quantities)."""
    s = str(text)
    for m in reversed(list(_CLOCK_TOKEN.finditer(s))):
        if m.group(2) is not None or m.group(3) is not None:
            s = s[: m.start()] + " " * (m.end() - m.start()) + s[m.end():]
    return s


def count_magnitudes(text: str) -> set[float]:
    return {float(a or b) for a, b in _COUNT_PHRASE.findall(str(text))}


def temperature_magnitudes(text: str) -> set[float]:
    return {float(x) for x in _TEMP_PHRASE.findall(str(text))}


def grounded_magnitudes(kind: str, text: str) -> set[float]:
    """Magnitudes a ``kind`` adjustment may take from this notice text."""
    if kind == "demand":
        return count_magnitudes(text)
    if kind == "weather":
        return temperature_magnitudes(text)
    return set()


def start_is_grounded(start: Any, text: str) -> bool:
    """``start`` is a whole HH:MM token whose time is written in the notice text."""
    m = _CLOCK_STRICT.match(str(start)) if isinstance(start, str) else None
    if m is None:
        return False
    return int(m.group(1)) * 60 + int(m.group(2)) in clock_tokens(text)


def _active(adj: dict[str, Any], minute: int) -> bool:
    try:
        a, b = parse_clock(adj["start"]), parse_clock(adj["end"])
    except Exception:
        return False
    return a <= minute < b


def sanitize_adjustment(adj: dict[str, Any], cfg: PlannerConfig) -> tuple[dict[str, Any] | None, str | None]:
    """Clip an adjustment to its declared bounds. Returns (clean, note) or (None, reason)."""
    kind = adj.get("kind")
    if kind not in ("demand", "weather"):
        return None, None
    try:
        parse_clock(str(adj["start"]))
        parse_clock(str(adj["end"]))
        delta = float(adj["delta"])
    except Exception:
        return None, "malformed adjustment"
    if not np.isfinite(delta):
        return None, "malformed adjustment"
    conf = adj.get("confidence")
    if conf not in CONFIDENCE_WEIGHT:
        return None, f"invalid confidence {conf!r}"
    allowed = {"source_id", "kind", "start", "end", "delta", "confidence", "evidence", "recheck_room"}
    extra = set(adj) - allowed
    if extra:
        return None, f"unexpected field(s) {sorted(extra)}"
    cap = cfg.forecast_delta_cap if kind == "demand" else cfg.weather_delta_cap_c
    note = None
    if abs(delta) > cap:
        note = f"delta {delta:g} clipped to ±{cap:g}"
        delta = float(np.clip(delta, -cap, cap))
    clean = dict(adj, delta=delta, confidence=conf)
    return clean, note


def _norm(t: str) -> str:
    return " ".join(str(t).lower().split())


def ground_adjustment(adj: dict[str, Any], notices: dict[str, Notice], screen: bool = True) -> str | None:
    """Deterministic grounding check on an interpreter output. Returns a reason, or None if grounded.

    The interpreter may not supply a number the text did not contain: a demand/weather
    adjustment must cite a released notice that is not instruction-like (only when ``screen``),
    quote evidence verbatim from it, take |delta| from a count phrase (temperature phrase for
    weather) in its text and anchor its start time to a whole clock token in its text.
    ``screen=False`` is for measuring the model alone (E5 live, ``model_only``).
    """
    kind = adj.get("kind")
    if kind not in ("demand", "weather"):
        return None
    sid = str(adj.get("source_id", ""))
    n = notices.get(sid)
    if n is None:
        return f"unknown source_id {sid!r}"
    if screen and looks_like_instruction(n.text):
        return "cited notice contains embedded instructions"
    ev = adj.get("evidence")
    if ev and _norm(ev) not in _norm(n.text):
        return "evidence is not a verbatim quote of the cited notice"
    try:
        mag = abs(float(adj["delta"]))
    except Exception:
        return "delta is not a number"
    if mag not in grounded_magnitudes(kind, n.text):
        what = "count phrase" if kind == "demand" else "temperature phrase"
        return f"delta {mag:g} is not a {what} of the notice text"
    if not start_is_grounded(adj.get("start"), n.text):
        return "start time is not a clock time written in the notice text"
    return None


_PLACES: frozenset[str] | None = None


def notice_places(text: str) -> frozenset[str]:
    """Places a notice names: the corpus venues (LT-A, Hall C, ...), room ids and buildings."""
    global _PLACES
    if _PLACES is None:
        import json
        from pathlib import Path

        corpus = Path(__file__).resolve().parent.parent / "data" / "notice_corpus.json"
        try:
            venues = json.loads(corpus.read_text()).get("places", [])
        except (OSError, ValueError):
            venues = []
        _PLACES = frozenset(str(v) for v in venues)
    low = str(text).lower()
    found = {v for v in _PLACES if re.search(rf"(?<![\w-]){re.escape(v.lower())}(?![\w-])", low)}
    found |= {r for r in ROOM_IDS if re.search(rf"\b{r.lower()}\b", low)}
    found |= {f"building {b}" for b in re.findall(r"\bbuilding\s+([ab])\b", low)}
    return frozenset(found)


def _same_event(a: dict[str, Any], b: dict[str, Any]) -> bool:
    pa, pb = set(a.get("_places") or ()), set(b.get("_places") or ())
    return not (pa and pb) or bool(pa & pb)


def resolve_conflicts(adjs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Same kind, overlapping windows, same place: keep the later notice (Table 10).

    Order is release time, then input order. Every replaced adjustment is reported; ``type`` is
    ``"contradiction"`` when the deltas differ and ``"duplicate"`` when they agree.
    """
    kept: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for a in sorted(adjs, key=lambda x: x.get("_release", 0)):
        clashes = [
            k for k in kept
            if k["kind"] == a["kind"] and _same_event(k, a) and not (
                parse_clock(a["end"]) <= parse_clock(k["start"]) or parse_clock(k["end"]) <= parse_clock(a["start"])
            )
        ]
        for k in clashes:
            conflicts.append({"earlier": k.get("source_id"), "later": a.get("source_id"), "kind": a["kind"],
                              "earlier_delta": k["delta"], "later_delta": a["delta"],
                              "type": "contradiction" if k["delta"] != a["delta"] else "duplicate",
                              "window": f"{a['start']}-{a['end']}", "resolution": "later notice kept"})
            kept.remove(k)
        kept.append(a)
    return kept, conflicts


class NoticeForecaster:
    """Constant forecast plus interpreter adjustments; interprets each notice once.

    ``screen`` (default: the interpreter's own ``screen`` setting, else True) controls whether
    grounding rejects adjustments that cite instruction-like notices."""

    def __init__(self, interpreter: Interpreter, cfg: PlannerConfig, screen: bool | None = None):
        self.interpreter = interpreter
        self.cfg = cfg
        self.screen = bool(getattr(interpreter, "screen", True)) if screen is None else bool(screen)
        self.name = f"notice:{interpreter.name}"
        self._seen: set[str] = set()
        self._adjs: list[dict[str, Any]] = []
        self.log: list[dict[str, Any]] = []
        self.recheck_requests: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._seen.clear()
        self._adjs.clear()
        self.log.clear()
        self.recheck_requests.clear()

    def update(self, snap: Snapshot) -> None:
        new = [n for n in snap.notices if n.notice_id not in self._seen]
        if not new:
            return
        for n in new:
            self._seen.add(n.notice_id)
        raw = self.interpreter.interpret(new, snap) or []
        release = {n.notice_id: n.release_min for n in new}
        released = {n.notice_id: n for n in snap.notices}
        have = {(str(a.get("source_id")), a.get("kind")) for a in self._adjs}
        for adj in raw:
            if not isinstance(adj, dict):
                self.log.append({"step": snap.step, "raw": adj, "applied": None, "note": "not an object"})
                continue
            if adj.get("kind") == "report" or adj.get("recheck_room"):
                self.recheck_requests.append({"step": snap.step, **adj})
            reason = ground_adjustment(adj, released, screen=self.screen)
            if reason is not None:
                self.log.append({"step": snap.step, "raw": adj, "applied": None, "note": f"ungrounded: {reason}"})
                continue
            clean, note = sanitize_adjustment(adj, self.cfg)
            if clean is not None:
                key = (str(clean.get("source_id")), clean.get("kind"))
                if key in have:
                    self.log.append({"step": snap.step, "raw": adj, "applied": None,
                                     "note": f"duplicate adjustment for {key[0]}/{key[1]} dropped"})
                    continue
                have.add(key)
            self.log.append({"step": snap.step, "raw": adj, "applied": clean, "note": note})
            if clean is not None:
                clean["_release"] = release.get(str(clean.get("source_id")), snap.clock_min)
                src = released.get(str(clean.get("source_id")))
                clean["_places"] = sorted(notice_places(src.text)) if src is not None else []
                self._adjs.append(clean)

    def forecast(self, snap: Snapshot, length: int) -> Forecast:
        self.update(snap)
        adjs, conflicts = resolve_conflicts(self._adjs)
        d = np.full(length, float(snap.demand))
        t = np.full(length + 1, float(snap.t_out))
        now = snap.clock_min
        for a in adjs:
            w = CONFIDENCE_WEIGHT[a["confidence"]]
            for k in range(length + 1):
                tau = now + k * STEP_MIN
                shift = w * a["delta"] * (float(_active(a, tau)) - float(_active(a, now)))
                if a["kind"] == "demand" and k < length:
                    d[k] += shift
                elif a["kind"] == "weather":
                    t[k] += shift
        cap = self.cfg.forecast_delta_cap
        d = np.clip(d, snap.demand - cap, snap.demand + cap)
        return Forecast(
            demand=np.maximum(0, np.rint(d)).astype(int),
            t_out=t,
            adjustments=[{k: v for k, v in a.items() if not k.startswith("_")} for a in adjs],
            conflicts=conflicts,
        )
