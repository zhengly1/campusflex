"""In-process message bus with MQTT-style topics and fault injection.

This is the fallback transport of Section 3.7: same topics and message schema as the
MQTT device layer, delivered synchronously inside one process. QoS-1-like semantics:
every published message is delivered at least once to every matching subscriber unless a
fault hook drops it. Fault hooks let the acceptance tests of Table 11 create duplicate,
out-of-order, dropped and delayed messages deterministically.

Payloads are JSON round-tripped on every delivery so handlers never share objects with
the publisher (as with a real broker).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Protocol

Handler = Callable[[str, dict], None]


def topic_matches(pattern: str, topic: str) -> bool:
    """MQTT wildcard matching: ``+`` matches one level, ``#`` matches the rest."""
    p, t = pattern.split("/"), topic.split("/")
    for i, part in enumerate(p):
        if part == "#":
            return True
        if i >= len(t):
            return False
        if part != "+" and part != t[i]:
            return False
    return len(p) == len(t)


class Bus(Protocol):
    """Transport used by ``BusDeviceLayer``."""

    name: str

    def subscribe(self, pattern: str, handler: Handler) -> None: ...
    def unsubscribe(self, pattern: str, handler: Handler) -> None: ...
    def publish(self, topic: str, payload: dict[str, Any]) -> None: ...
    def pump(self, until: Callable[[], bool] | None = None, timeout: float = 0.0) -> None: ...
    def close(self) -> None: ...


@dataclass
class _Fault:
    kind: str            # "drop" | "duplicate" | "hold" | "delay"
    pattern: str
    remaining: int
    ticks: int = 0


@dataclass
class _Delayed:
    due_tick: int
    topic: str
    payload: str


@dataclass
class BusStats:
    published: int = 0
    delivered: int = 0
    dropped: int = 0
    duplicated: int = 0
    held: int = 0
    delayed: int = 0

    def to_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


class InProcessBus:
    name = "inprocess"

    def __init__(self) -> None:
        self._subs: list[tuple[str, Handler]] = []
        self._faults: list[_Fault] = []
        self._held: list[tuple[str, str]] = []
        self._delayed: list[_Delayed] = []
        self.tick = 0
        self.stats = BusStats()
        self.closed = False

    # ------------------------------------------------------------- pub/sub
    def subscribe(self, pattern: str, handler: Handler) -> None:
        self._subs.append((pattern, handler))

    def unsubscribe(self, pattern: str, handler: Handler) -> None:
        self._subs = [(p, h) for p, h in self._subs if not (p == pattern and h == handler)]

    def publish(self, topic: str, payload: dict[str, Any]) -> None:
        if self.closed:
            raise RuntimeError("bus is closed")
        self.stats.published += 1
        wire = json.dumps(payload)
        fault = self._take_fault(topic)
        if fault is None:
            self._deliver(topic, wire)
        elif fault.kind == "drop":
            self.stats.dropped += 1
        elif fault.kind == "duplicate":
            self.stats.duplicated += 1
            self._deliver(topic, wire)
            self._deliver(topic, wire)
        elif fault.kind == "hold":
            self.stats.held += 1
            self._held.append((topic, wire))
        elif fault.kind == "delay":
            self.stats.delayed += 1
            self._delayed.append(_Delayed(self.tick + max(1, fault.ticks), topic, wire))

    def _deliver(self, topic: str, wire: str) -> None:
        for pattern, handler in list(self._subs):
            if topic_matches(pattern, topic):
                self.stats.delivered += 1
                handler(topic, json.loads(wire))

    def pump(self, until: Callable[[], bool] | None = None, timeout: float = 0.0) -> None:
        """Delivery is synchronous; nothing is pending except held/delayed messages."""
        return None

    def close(self) -> None:
        self._subs.clear()
        self.closed = True

    # -------------------------------------------------------- fault hooks
    def _take_fault(self, topic: str) -> _Fault | None:
        for f in self._faults:
            if f.remaining > 0 and topic_matches(f.pattern, topic):
                f.remaining -= 1
                return f
        return None

    def _add(self, kind: str, pattern: str, n: int, ticks: int = 0) -> None:
        self._faults.append(_Fault(kind, pattern, n, ticks))

    def drop_next(self, pattern: str, n: int = 1) -> None:
        """Lose the next ``n`` messages whose topic matches ``pattern``."""
        self._add("drop", pattern, n)

    def duplicate_next(self, pattern: str, n: int = 1) -> None:
        """Deliver each of the next ``n`` matching messages twice."""
        self._add("duplicate", pattern, n)

    def hold_next(self, pattern: str, n: int = 1) -> None:
        """Hold the next ``n`` matching messages until ``release_held`` (for reordering)."""
        self._add("hold", pattern, n)

    def delay_next(self, pattern: str, n: int = 1, ticks: int = 1) -> None:
        """Deliver the next ``n`` matching messages ``ticks`` simulated intervals later."""
        self._add("delay", pattern, n, ticks)

    def clear_faults(self) -> None:
        self._faults.clear()

    def release_held(self, reverse: bool = False) -> int:
        """Deliver held messages; ``reverse=True`` delivers them out of order."""
        held, self._held = self._held, []
        for topic, wire in (reversed(held) if reverse else held):
            self._deliver(topic, wire)
        return len(held)

    def advance(self, ticks: int = 1) -> int:
        """Advance simulated time by ``ticks`` intervals and deliver messages that are due."""
        self.tick += ticks
        due = [d for d in self._delayed if d.due_tick <= self.tick]
        self._delayed = [d for d in self._delayed if d.due_tick > self.tick]
        for d in due:
            self._deliver(d.topic, d.payload)
        return len(due)
