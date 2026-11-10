"""Classroom devices of the smart extension and their firmware.

Every room has six controllable devices:

=========  =============================================  ==========================
device     value                                          notes
=========  =============================================  ==========================
ac         ``{"setpoint": float | None, "fan": 0..3}``    None or fan 0 → AC off
light      ``{"level": 0..100}`` (steps of 10)            dimmable LED, % of P_light
vent       ``{"level": 0..3}``                            fresh air 4/7/10 L/s per seat, ERV
curtain    ``{"open_pct": 0..100}`` (steps of 10)         daylight and solar gain
projector  ``{"on": bool}``
sockets    ``{"on": bool}``                               laptop charging / standby
=========  =============================================  ==========================

The firmware validates each command, applies it idempotently (a replayed ``cmd_id``
returns the original ack), refuses AC cooling when the unit is offline and refuses
commands older than the last one applied to the same device.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..config import ROOM_BY_ID, ROOM_IDS

DEVICES = ("ac", "light", "vent", "curtain", "projector", "sockets")
USER_DEVICES = ("ac", "light", "curtain", "projector")       # what occupants adjust by hand

SETPOINT_RANGE = (22.0, 28.0)
FAN_LEVELS = (0, 1, 2, 3)
FAN_CAPACITY = {0: 0.0, 1: 0.75, 2: 0.9, 3: 1.0}             # share of rated cooling delivered
FAN_KW = {0: 0.0, 1: 0.04, 2: 0.07, 3: 0.11}
VENT_LEVELS = (0, 1, 2, 3)
VENT_LPS_PER_SEAT = {0: 0.0, 1: 4.0, 2: 7.0, 3: 10.0}        # fresh air per seat (L/s)
VENT_KW_PER_SEAT = {0: 0.0, 1: 0.0007, 2: 0.0015, 3: 0.0027}
PROJECTOR_KW = 0.30
PROJECTOR_STANDBY_KW = 0.005
SOCKET_BASE_KW = 0.03                                         # chargers and screens on the strip
SOCKET_PER_PERSON_KW = 0.006                                  # mean laptop charging share


def vent_flow_m3s(room: str, level: int) -> float:
    return ROOM_BY_ID[room].seats * VENT_LPS_PER_SEAT[level] / 1000.0


def vent_kw(room: str, level: int) -> float:
    return ROOM_BY_ID[room].seats * VENT_KW_PER_SEAT[level]


@dataclass
class DeviceState:
    """Applied state of one room's devices."""

    ac_setpoint: float | None = None
    ac_fan: int = 0
    light: int = 0
    vent: int = 0
    curtain: int = 50
    projector: bool = False
    sockets: bool = False

    def ac_on(self) -> bool:
        return self.ac_setpoint is not None and self.ac_fan > 0

    def value(self, device: str) -> dict[str, Any]:
        return {
            "ac": {"setpoint": self.ac_setpoint, "fan": self.ac_fan},
            "light": {"level": self.light},
            "vent": {"level": self.vent},
            "curtain": {"open_pct": self.curtain},
            "projector": {"on": self.projector},
            "sockets": {"on": self.sockets},
        }[device]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SmartCommand:
    cmd_id: str                  # idempotency key
    room: str
    device: str
    value: dict[str, Any]
    issued_min: int
    seq: int = 0
    source: str = "scheduler"    # "scheduler" | "user"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SmartCommand":
        return cls(**{k: d[k] for k in ("cmd_id", "room", "device", "value", "issued_min")},
                   seq=int(d.get("seq", 0)), source=str(d.get("source", "scheduler")))


@dataclass
class SmartAck:
    cmd_id: str
    room: str
    device: str
    ok: bool
    reason: str = ""
    applied: dict[str, Any] = field(default_factory=dict)
    duplicate: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SmartAck":
        return cls(**{k: d[k] for k in ("cmd_id", "room", "device", "ok")}, reason=d.get("reason", ""),
                   applied=d.get("applied") or {}, duplicate=bool(d.get("duplicate", False)))


def validate_value(device: str, value: Any) -> str | None:
    """None when ``value`` is a valid setting of ``device``, else the reason."""
    if device not in DEVICES:
        return f"unknown device {device!r}"
    if not isinstance(value, dict):
        return "value must be an object"
    if device == "ac":
        sp, fan = value.get("setpoint"), value.get("fan")
        if fan not in FAN_LEVELS:
            return f"fan must be one of {FAN_LEVELS}"
        if sp is not None and (isinstance(sp, bool) or not isinstance(sp, (int, float))
                               or not SETPOINT_RANGE[0] <= float(sp) <= SETPOINT_RANGE[1]):
            return f"setpoint must be None or within {SETPOINT_RANGE}"
    elif device in ("light", "curtain"):
        key = "level" if device == "light" else "open_pct"
        v = value.get(key)
        if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 100 or v % 10:
            return f"{key} must be 0..100 in steps of 10"
    elif device == "vent":
        if value.get("level") not in VENT_LEVELS:
            return f"level must be one of {VENT_LEVELS}"
    elif not isinstance(value.get("on"), bool):
        return "on must be true or false"
    return None


class SmartRoomFirmware:
    """Applies commands to one room's ``DeviceState``; ``health()`` is the AC unit's health."""

    def __init__(self, room: str, state: DeviceState, health=lambda: 1.0):
        if room not in ROOM_IDS:
            raise ValueError(f"unknown room {room!r}")
        self.room = room
        self.state = state
        self.health = health
        self._done: dict[str, SmartAck] = {}
        self._last_seq: dict[str, int] = {}

    def handle(self, cmd: SmartCommand) -> SmartAck:
        if cmd.cmd_id in self._done:
            a = self._done[cmd.cmd_id]
            return SmartAck(a.cmd_id, a.room, a.device, a.ok, a.reason, dict(a.applied), duplicate=True)
        ack = self._apply(cmd)
        self._done[cmd.cmd_id] = ack
        return ack

    def _refuse(self, cmd: SmartCommand, reason: str) -> SmartAck:
        return SmartAck(cmd.cmd_id, self.room, cmd.device, False, reason, self.state.value(cmd.device)
                        if cmd.device in DEVICES else {})

    def _apply(self, cmd: SmartCommand) -> SmartAck:
        if cmd.room != self.room:
            return self._refuse(cmd, f"command for {cmd.room} sent to {self.room}")
        bad = validate_value(cmd.device, cmd.value)
        if bad:
            return self._refuse(cmd, bad)
        if cmd.seq and cmd.seq <= self._last_seq.get(cmd.device, 0):
            return self._refuse(cmd, "out_of_order")
        s, v = self.state, cmd.value
        if cmd.device == "ac":
            cooling = v["setpoint"] is not None and v["fan"] > 0
            if cooling and self.health() <= 0:
                return self._refuse(cmd, "ac_offline")
            s.ac_setpoint = None if v["setpoint"] is None else round(float(v["setpoint"]), 2)
            s.ac_fan = int(v["fan"]) if s.ac_setpoint is not None else 0
        elif cmd.device == "light":
            s.light = int(v["level"])
        elif cmd.device == "vent":
            s.vent = int(v["level"])
        elif cmd.device == "curtain":
            s.curtain = int(v["open_pct"])
        elif cmd.device == "projector":
            s.projector = bool(v["on"])
        else:
            s.sockets = bool(v["on"])
        if cmd.seq:
            self._last_seq[cmd.device] = cmd.seq
        return SmartAck(cmd.cmd_id, self.room, cmd.device, True, "", s.value(cmd.device))
