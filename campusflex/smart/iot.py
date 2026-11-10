"""IoT transport of the smart extension (MQTT topics, or the in-process bus).

Topics (one subscription per room on each side)::

    campus/<room>/<device>/cmd     gateway or wall panel → device firmware
    campus/<room>/<device>/ack     firmware → gateway (``source`` says who issued it)
    campus/<room>/sensors          room sensors → gateway, every tick

Commands carry an idempotency key and a per-device sequence number; the gateway retries a
command whose ack did not arrive with the same key, so a replay is never applied twice.
Every payload carries the run id; messages of another run on the same broker are ignored.
"""
from __future__ import annotations

import uuid
from typing import Any

from ..config import ROOM_IDS
from .devices import DEVICES, DeviceState, SmartAck, SmartCommand, SmartRoomFirmware
from .physics import SmartPlant


class SmartIoT:
    def __init__(self, bus: Any, plant: SmartPlant, prefix: str = "campus", max_retries: int = 2,
                 ack_timeout_s: float = 1.0, owns_bus: bool = False):
        self.bus, self.plant, self.prefix = bus, plant, prefix
        self.max_retries, self.ack_timeout_s, self.owns_bus = max_retries, ack_timeout_s, owns_bus
        self.run_id = uuid.uuid4().hex[:12]
        self.firmware = {r: SmartRoomFirmware(r, plant.devices[r], health=lambda r=r: plant.health[r])
                         for r in ROOM_IDS}
        self.applied = {r: DeviceState(**plant.devices[r].to_dict()) for r in ROOM_IDS}   # gateway's view
        self.acks: dict[str, SmartAck] = {}
        self.telemetry: dict[str, dict[str, Any]] = {}
        self._seq: dict[tuple[str, str], int] = {}
        self.stats = {"cmds_sent": 0, "retries": 0, "acks": 0, "refused": 0, "missing_acks": 0,
                      "telemetry": 0, "foreign": 0}
        self._subs: list[tuple[str, Any]] = []
        for r in ROOM_IDS:
            self._sub(f"{prefix}/{r}/+/cmd", self._on_cmd)
        self._sub(f"{prefix}/+/+/ack", self._on_ack)
        self._sub(f"{prefix}/+/sensors", self._on_sensors)

    def _sub(self, pattern: str, handler: Any) -> None:
        self.bus.subscribe(pattern, handler)
        self._subs.append((pattern, handler))

    def close(self) -> None:
        for p, h in self._subs:
            try:
                self.bus.unsubscribe(p, h)
            except Exception:  # pragma: no cover - transport already gone
                pass
        if self.owns_bus:
            self.bus.close()

    # ------------------------------------------------------------- device side
    def _foreign(self, payload: dict[str, Any]) -> bool:
        if payload.get("run") != self.run_id:
            self.stats["foreign"] += 1
            return True
        return False

    def _on_cmd(self, topic: str, payload: dict[str, Any]) -> None:
        if self._foreign(payload):
            return
        try:
            cmd = SmartCommand.from_dict(payload)
        except (KeyError, TypeError):
            return
        ack = self.firmware[cmd.room].handle(cmd)
        self.bus.publish(f"{self.prefix}/{cmd.room}/{cmd.device}/ack",
                         {"run": self.run_id, "source": cmd.source, **ack.to_dict()})

    # ------------------------------------------------------------ gateway side
    def _on_ack(self, topic: str, payload: dict[str, Any]) -> None:
        if self._foreign(payload):
            return
        ack = SmartAck.from_dict(payload)
        if ack.cmd_id in self.acks:
            return
        self.acks[ack.cmd_id] = ack
        self.stats["acks"] += 1
        if not ack.ok:
            self.stats["refused"] += 1
        if ack.device in DEVICES and ack.applied:
            mirror = self.applied[ack.room]
            v = ack.applied
            if ack.device == "ac":
                mirror.ac_setpoint, mirror.ac_fan = v.get("setpoint"), int(v.get("fan", 0))
            elif ack.device == "light":
                mirror.light = int(v["level"])
            elif ack.device == "vent":
                mirror.vent = int(v["level"])
            elif ack.device == "curtain":
                mirror.curtain = int(v["open_pct"])
            elif ack.device == "projector":
                mirror.projector = bool(v["on"])
            else:
                mirror.sockets = bool(v["on"])

    def _on_sensors(self, topic: str, payload: dict[str, Any]) -> None:
        if self._foreign(payload):
            return
        self.telemetry[payload["room"]] = payload
        self.stats["telemetry"] += 1

    def publish_sensors(self, msgs: list[dict[str, Any]]) -> None:
        for m in msgs:
            self.bus.publish(f"{self.prefix}/{m['room']}/sensors", {"run": self.run_id, **m})
        want = {m["room"]: m["minute"] for m in msgs}
        self.bus.pump(until=lambda: all(self.telemetry.get(r, {}).get("minute") == t for r, t in want.items()),
                      timeout=self.ack_timeout_s)

    def command(self, room: str, device: str, value: dict[str, Any], minute: int,
                source: str = "scheduler") -> SmartAck | None:
        key = (room, device)
        self._seq[key] = self._seq.get(key, 0) + 1
        cmd = SmartCommand(f"{room}/{device}/{self.run_id}/{self._seq[key]}", room, device, value, minute,
                           seq=self._seq[key], source=source)
        for attempt in range(1 + self.max_retries):
            if attempt:
                self.stats["retries"] += 1
            self.stats["cmds_sent"] += 1
            self.bus.publish(f"{self.prefix}/{room}/{device}/cmd", {"run": self.run_id, **cmd.to_dict()})
            self.bus.pump(until=lambda: cmd.cmd_id in self.acks, timeout=self.ack_timeout_s)
            if cmd.cmd_id in self.acks:
                return self.acks[cmd.cmd_id]
        self.stats["missing_acks"] += 1
        return None
