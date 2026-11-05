"""Number-matching faithfulness checker (L4, L5 report; experiment E4).

Every number in text shown to a person must match a value that a tool logged. The checker
extracts numbers, matches each against the logged values at the precision it was written
with, and removes and flags any number that does not match before display.
Clock times (19:40) and room identifiers (B201) are not treated as quantities. A logged
fraction may be written as a percentage only when "%"/"percent" follows the number; number
words followed by a unit ("fifteen percent", "two moves") are checked like digits.

Units: every logged value takes a quantity class from its field name (``kwh``/``E`` energy,
``mean_dev_c``/``dev_mean`` °C, ``moves``/``M``, ``K`` switches, ``*_person_min``/``U``,
``*_min`` minutes, ``rank``/``score_J`` and review counters "other"). A number written with
a unit ("6 kWh", "2 room switches", "12 students") matches only values of a class that unit
can denote (a move count is a number of students); values whose field has no class
(``extra_allowed``, parameter values) match any unit. A number without a unit is matched by
value alone. A number whose value exists only under another unit is flagged and listed in
``unit_mismatch``.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# A number not glued to letters (room ids such as B201) and not part of a clock time;
# scientific notation (1e3, 2.5E-1) is one number.
_NUM = re.compile(r"(?<![A-Za-z0-9.:])([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?![\d:]|[A-Za-z]\d)")
_CLOCK = re.compile(r"\b\d{1,2}:\d{2}\b")
_ROOM = re.compile(r"\b[AB]\d{3}\b")
# "Option 2" is a label; "Option 1.5 kWh" is not (the label may not swallow a decimal).
_OPTION = re.compile(r"\boption\s*#?\s*(\d+)(?![.,]\d)\b", re.I)
_PERCENT = re.compile(r"\s*(?:%|percent\b|per\s*cent\b|pct\b)", re.I)
_SMALL = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
          "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
         "ninety": 90}
_WORD_VALUE = {w: i for i, w in enumerate(_SMALL)} | dict(_TENS) | {"hundred": 100}
# A number written as a word is a quantity when a unit follows it ("fifteen percent", "two moves");
# "one of the rooms" is not checked.
_WORD_NUM = re.compile(
    r"\b((?:" + "|".join(_TENS) + r")(?:[-\s](?:one|two|three|four|five|six|seven|eight|nine))?|"
    + "|".join(sorted(_WORD_VALUE, key=len, reverse=True)) + r")"
    r"(?=\s*(?:%|percent\b|per\s*cent\b|kwh\b|kw\b|degrees?\b|°|students?\b|moves?\b|relocations?\b|"
    r"minutes?\b|seats?\b|rooms?\b|switch(?:es)?\b|people\b|person-minutes\b))",
    re.I,
)


# Unit written right after a number, optionally after one qualifier ("3 room switches",
# "12 student moves", "40 unserved person-minutes"); the group name is the quantity class.
_UNIT = re.compile(
    r"\s*(?:(?:unserved|served|extra|additional|more|fewer|less|student|occupied|room)\s+)?(?:"
    r"(?P<person_min>person[-\s]?min(?:ute)?s?\b)|(?P<energy>kwh\b)|(?P<power>kw\b)|"
    r"(?P<temp>°\s*c?|℃|degrees?\b|deg\b)|(?P<switches>switch(?:es)?\b)|"
    r"(?P<moves>(?:moves?|relocations?)\b)|(?P<people>(?:students?|people|persons?|seats?)\b)|"
    r"(?P<minutes>(?:minutes?|mins?)\b))",
    re.I,
)
# Value classes a written unit may denote (a move count is a number of students).
_ACCEPTS: dict[str, frozenset[str]] = {
    "energy": frozenset({"energy"}), "power": frozenset({"power"}), "temp": frozenset({"temp"}),
    "switches": frozenset({"switches"}), "moves": frozenset({"moves"}),
    "people": frozenset({"people", "moves"}), "person_min": frozenset({"person_min"}),
    "minutes": frozenset({"minutes", "person_min"}),
}
# Fields that are known not to be a physical quantity (ranks, scores, review counters).
_OTHER = frozenset({"rank", "score", "score_j", "score_u", "omega", "v", "index", "option", "n", "rounds",
                    "proposals", "accepted", "seed", "step", "version", "tuning", "confirmation", "baseline",
                    "mean_diff", "lo", "hi", "base_mean", "cand_mean"})
_PEOPLE = frozenset({"students", "demand", "served", "unserved", "counts", "capacity", "seats", "people",
                     "occupancy"})


def value_class(key: str | None) -> str | None:
    """Quantity class of a logged value from the name of the field holding it (None: unknown)."""
    if key is None:
        return None
    k = str(key).lower()
    if "person_min" in k or k == "u":
        return "person_min"
    if "kwh" in k or k in ("e", "energy", "pred_e", "pred_first_e", "first_e"):
        return "energy"
    if k == "kw" or k.endswith("_kw"):
        return "power"
    if "dev" in k or k.endswith("_c") or "temp" in k or k in ("t_end", "t_start", "t_out"):
        return "temp"
    if "switch" in k or k == "k":
        return "switches"
    if "move" in k or k in ("m", "count"):
        return "moves"
    if k.endswith("_min") or "minute" in k:
        return "minutes"
    if k in _PEOPLE:
        return "people"
    if k in _OTHER:
        return "other"
    return None


def unit_at(text: str, pos: int) -> str | None:
    """Quantity class of the unit written at ``text[pos:]`` (None: no unit)."""
    m = _UNIT.match(text, pos)
    return None if m is None else m.lastgroup


def _word_value(word: str) -> float:
    parts = re.split(r"[-\s]+", word.lower())
    return float(sum(_WORD_VALUE[p] for p in parts))


@dataclass
class FaithfulnessResult:
    numbers: list[str] = field(default_factory=list)
    matched: list[str] = field(default_factory=list)
    flagged: list[str] = field(default_factory=list)
    unit_mismatch: list[str] = field(default_factory=list)   # flagged: value logged only under another unit
    cleaned_text: str = ""

    @property
    def share_matched(self) -> float:
        return 1.0 if not self.numbers else len(self.matched) / len(self.numbers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "numbers": self.numbers,
            "matched": self.matched,
            "flagged": self.flagged,
            "unit_mismatch": self.unit_mismatch,
            "share_matched": round(self.share_matched, 4),
        }


def flatten_values(obj: Any) -> list[float]:
    """Collect every numeric leaf from a nested tool-output structure."""
    out: list[float] = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        if math.isfinite(float(obj)):
            out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            out += flatten_values(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += flatten_values(v)
    return out


def flatten_tagged(obj: Any, key: str | None = None) -> list[tuple[float, str | None]]:
    """Every numeric leaf with the quantity class of the field holding it (list items inherit
    the field name of the list)."""
    out: list[tuple[float, str | None]] = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        if math.isfinite(float(obj)):
            out.append((float(obj), value_class(key)))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out += flatten_tagged(v, str(k))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += flatten_tagged(v, key)
    return out


def extract_numbers(text: str) -> list[tuple[str, int, int]]:
    masked = _CLOCK.sub(lambda m: "#" * len(m.group()), text)
    masked = _ROOM.sub(lambda m: "#" * len(m.group()), masked)
    return [(m.group(1), m.start(1), m.end(1)) for m in _NUM.finditer(masked)]


def _tolerance(token: str) -> float:
    """Half a unit of the last digit written (1.25 → 0.005, 1e3 → 500, 2.5e-1 → 0.05)."""
    mant, _, exp = token.lower().partition("e")
    decimals = len(mant.split(".")[1]) if "." in mant else 0
    return 0.5 * 10 ** (int(exp or 0) - decimals) + 1e-9


def _matches(token: str, values: Iterable[float], percent: bool = False, x: float | None = None) -> bool:
    """``token`` written at its precision equals a logged value, or that value without its sign;
    a percentage ("12 %") may also render a logged fraction (0.12), and only a percentage may."""
    tol = _tolerance(token) if x is None else 0.5 + 1e-9        # a number word is a whole number
    x = float(token) if x is None else x
    for v in values:
        cands = (v, abs(v), 100.0 * v, 100.0 * abs(v)) if percent else (v, abs(v))
        if any(abs(c - x) <= tol for c in cands):
            return True
    return False


def check(text: str, logged: Any, extra_allowed: Iterable[float] = (),
          option_labels: Iterable[int] | None = None) -> FaithfulnessResult:
    """Match every number in ``text`` against the numeric leaves of ``logged``.

    ``option_labels``: when given, "Option N" is an identifier, not a quantity: a label of an
    option that exists is ignored, any other "Option N" is flagged. The option numbers are
    *not* allowed as values anywhere else in the text (so "saves 1 kWh" cannot hide behind
    "Option 1")."""
    tagged = flatten_tagged(logged) + [(float(v), None) for v in extra_allowed]
    values = [v for v, _ in tagged]
    res = FaithfulnessResult()
    items: list[tuple[int, int, str, bool]] = []          # (start, end, token, matched)
    mismatch: set[tuple[int, int]] = set()

    def judge(tok: str, a: int, b: int, x: float | None = None) -> bool:
        percent = bool(_PERCENT.match(masked, b))
        unit = None if percent else unit_at(masked, b)
        if unit is None:
            return _matches(tok, values, percent=percent, x=x)
        ok = _matches(tok, [v for v, c in tagged if c is None or c in _ACCEPTS[unit]], x=x)
        if not ok and _matches(tok, values, x=x):
            mismatch.add((a, b))
        return ok

    label_spans: list[tuple[int, int]] = []
    if option_labels is not None:
        labels = {int(x) for x in option_labels}
        for m in _OPTION.finditer(text):
            label_spans.append((m.start(1), m.end(1)))
            if int(m.group(1)) not in labels:
                items.append((m.start(1), m.end(1), m.group(1), False))
    masked = text
    for a, b in label_spans:
        masked = masked[:a] + "#" * (b - a) + masked[b:]
    for tok, a, b in extract_numbers(masked):
        items.append((a, b, tok, judge(tok, a, b)))
    for m in _WORD_NUM.finditer(masked):
        items.append((m.start(1), m.end(1), m.group(1), judge(m.group(1), m.start(1), m.end(1), _word_value(m.group(1)))))
    pieces, last = [], 0
    for a, b, tok, ok in sorted(items):
        res.numbers.append(tok)
        if ok:
            res.matched.append(tok)
        else:
            res.flagged.append(tok)
            if (a, b) in mismatch:
                res.unit_mismatch.append(tok)
            pieces.append(text[last:a] + "[unverified]")
            last = b
    pieces.append(text[last:])
    res.cleaned_text = "".join(pieces)
    return res
