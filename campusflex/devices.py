"""Device layer contract (Section 2.4 IoT row, Table 11).

The *plant* is the simulated physical campus. Each room has a ``RoomDevice`` (firmware)
that applies commands idempotently, refuses AC commands to an offline unit, and reports
timestamped telemetry with sequence numbers. A ``DeviceLayer`` carries commands, acks and
telemetry between the controller and the devices:

* ``DirectDeviceLayer`` – in-process function calls (no transport); used by experiments.
* ``campusflex.iot.BusDeviceLayer`` – the same messages over MQTT topics
  ``campus/<room>/telemetry|cmd|ack`` or the in-process bus fallback.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np

from .config import N_ROOMS, ROOM_IDS, ROOMS


@dataclass
class Telemetry:
    room: str
    seq: int
    ts_min: int                 # simulated minute of day of the measurement
    temp_c: float
    health: float
    occupancy: int
    open: bool
    setpoint: float | None = None   # applied AC setpoint (None → AC off)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Telemetry":
        return cls(**{k: d[k] for k in ("room", "seq", "ts_min", "temp_c", "health", "occupancy", "open")},
                   setpoint=d.get("setpoint"))


@dataclass
class Command:
    cmd_id: str                 # idempotency key (per room)
    room: str
    open: bool
    setpoint: float | None      # None → AC off
    issued_min: int
    seq: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Command":
        return cls(**{k: d[k] for k in ("cmd_id", "room", "open", "setpoint", "issued_min", "seq")})


@dataclass
class Ack:
    cmd_id: str
    room: str
    ok: bool
    open: bool
    setpoint: float | None
    ts_min: int
    reason: str = ""
    duplicate: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Ack":
        return cls(**{k: d[k] for k in ("cmd_id", "room", "ok", "open", "setpoint", "ts_min", "reason", "duplicate")})


@dataclass
class Plant:
    """True physical state of the six rooms (owned by the environment)."""

    T: np.ndarray
    health: np.ndarray = field(default_factory=lambda: np.ones(N_ROOMS))
    open: np.ndarray = field(default_factory=lambda: np.zeros(N_ROOMS, dtype=bool))
    setpoint: np.ndarray = field(default_factory=lambda: np.full(N_ROOMS, np.nan))
    occupancy: np.ndarray = field(default_factory=lambda: np.zeros(N_ROOMS, dtype=int))

    @classmethod
    def initial(cls) -> "Plant":
        return cls(T=np.array([r.T0 for r in ROOMS], dtype=float))


class RoomDevice:
    """Simulated room controller firmware.

    Applies each ``cmd_id`` once (replays are re-acked) and refuses a command issued earlier
    than the last one it applied (a delayed message must not undo a newer decision).
    """

    def __init__(self, room: str, plant: Plant):
        self.room = room
        self.i = ROOM_IDS.index(room)
        self.plant = plant
        self.seq = 0
        self._acks: dict[str, Ack] = {}
        self._last_issued = -1

    def handle(self, cmd: Command, now_min: int) -> Ack:
        if cmd.cmd_id in self._acks:                       # idempotent replay
            prev = self._acks[cmd.cmd_id]
            return Ack(**{**asdict(prev), "duplicate": True})
        p, i = self.plant, self.i
        if cmd.issued_min < self._last_issued:             # out-of-order delivery
            sp_now = None if np.isnan(p.setpoint[i]) else float(p.setpoint[i])
            ack = Ack(cmd.cmd_id, self.room, ok=False, open=bool(p.open[i]), setpoint=sp_now,
                      ts_min=now_min, reason="out_of_order")
            self._acks[cmd.cmd_id] = ack
            return ack
        self._last_issued = cmd.issued_min
        reason = ""
        sp = cmd.setpoint
        if sp is not None and p.health[i] <= 0:
            sp, reason = None, "ac_offline"                 # refuse cooling command, still switch lights
        p.open[i] = bool(cmd.open)
        p.setpoint[i] = np.nan if sp is None else float(sp)
        ack = Ack(cmd.cmd_id, self.room, ok=(reason == ""), open=bool(cmd.open), setpoint=sp,
                  ts_min=now_min, reason=reason)
        self._acks[cmd.cmd_id] = ack
        return ack

    def read(self, now_min: int) -> Telemetry:
        self.seq += 1
        p, i = self.plant, self.i
        sp = None if np.isnan(p.setpoint[i]) else float(p.setpoint[i])
        return Telemetry(self.room, self.seq, now_min, float(p.T[i]), float(p.health[i]),
                         int(p.occupancy[i]), bool(p.open[i]), sp)


class DeviceLayer(Protocol):
    name: str

    def bind(self, plant: Plant) -> None: ...
    def publish(self, now_min: int) -> None: ...
    def latest(self) -> dict[str, Telemetry]: ...
    def command(self, cmds: list[Command], now_min: int) -> dict[str, Ack | None]: ...
    def close(self) -> None: ...


class DirectDeviceLayer:
    """No transport: commands and telemetry are function calls."""

    name = "direct"

    def __init__(self) -> None:
        self.devices: dict[str, RoomDevice] = {}
        self._latest: dict[str, Telemetry] = {}

    def bind(self, plant: Plant) -> None:
        self.devices = {r: RoomDevice(r, plant) for r in ROOM_IDS}
        self._latest = {}

    def publish(self, now_min: int) -> None:
        for r, d in self.devices.items():
            self._latest[r] = d.read(now_min)

    def latest(self) -> dict[str, Telemetry]:
        return dict(self._latest)

    def command(self, cmds: list[Command], now_min: int) -> dict[str, Ack | None]:
        return {c.room: self.devices[c.room].handle(c, now_min) for c in cmds}

    def close(self) -> None:
        pass
