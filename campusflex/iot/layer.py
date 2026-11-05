"""Device layer over a message bus (Table 11, IoT row).

Topics (QoS 1, JSON):
    campus/<room>/telemetry   device → gateway   Telemetry + seq + ts_min
    campus/<room>/cmd         gateway → device   Command (cmd_id is the idempotency key) + seq
    campus/<room>/ack         device → gateway   Ack + seq

Device side: one ``RoomDevice`` per room applies each cmd_id once and re-acks replays.
Gateway side: dedups telemetry by (room, seq), ignores out-of-order (older seq) readings,
keeps the latest reading per room, matches acks by cmd_id, and retries a command with the
SAME cmd_id when its ack is missing; after the retries it reports ``None`` (no ack), so the
environment leaves its ledger and believed state for that room unchanged. Staleness is
judged by the environment from ``Telemetry.ts_min`` (older than two intervals).
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from ..config import ROOM_IDS, STEP_MIN
from ..devices import Ack, Command, DirectDeviceLayer, Plant, RoomDevice, Telemetry
from .bus import Bus, InProcessBus

log = logging.getLogger(__name__)


@dataclass
class _DeviceEndpoint:
    """Device-side cmd handler for one room (a callable object, so the layer stays deep-copyable)."""

    layer: "BusDeviceLayer"
    device: RoomDevice

    def __call__(self, topic: str, payload: dict[str, Any]) -> None:
        self.layer._on_cmd(self.device, payload)


class BusDeviceLayer:
    def __init__(
        self,
        bus: Bus,
        prefix: str = "campus",
        max_retries: int = 2,
        ack_timeout_s: float = 1.0,
        telemetry_timeout_s: float = 0.5,
        owns_bus: bool = False,
    ):
        self.bus = bus
        self.owns_bus = owns_bus           # created by make_device_layer: closed with the layer
        self.prefix = prefix
        self.max_retries = max_retries
        self.ack_timeout_s = ack_timeout_s
        self.telemetry_timeout_s = telemetry_timeout_s
        self.name = f"bus:{getattr(bus, 'name', 'bus')}"
        # Every message carries the run id; messages of another run sharing the broker and
        # topic prefix are ignored by both the devices and the gateway.
        self.run_id = uuid.uuid4().hex[:12]
        self.fallback = False              # set by make_device_layer when MQTT was unavailable
        self.fallback_reason = ""
        self.devices: dict[str, RoomDevice] = {}
        self._handlers: list[tuple[str, Any]] = []
        self._reset_gateway()

    # ------------------------------------------------------------------ helpers
    def topic(self, room: str, kind: str) -> str:
        return f"{self.prefix}/{room}/{kind}"

    def _reset_gateway(self) -> None:
        self._now = 0
        self._last_tick_min: int | None = None
        self._latest: dict[str, Telemetry] = {}
        self._last_seq: dict[str, int] = {}
        self._acks: dict[str, Ack] = {}
        self._sent: set[str] = set()
        self._cmd_seq = 0
        self._ack_seq: dict[str, int] = {}
        self.stats = {
            "telemetry_received": 0,
            "telemetry_duplicates": 0,
            "telemetry_out_of_order": 0,
            "cmds_sent": 0,
            "cmd_retries": 0,
            "cmd_replays_at_device": 0,
            "acks_received": 0,
            "ack_duplicates": 0,
            "missing_acks": 0,
            "unknown_acks": 0,
            "foreign_msgs": 0,
        }

    def _sub(self, pattern: str, handler: Any) -> None:
        self.bus.subscribe(pattern, handler)
        self._handlers.append((pattern, handler))

    def _unsub_all(self) -> None:
        for pattern, handler in self._handlers:
            try:
                self.bus.unsubscribe(pattern, handler)
            except Exception:  # pragma: no cover - transport already gone
                pass
        self._handlers.clear()

    # --------------------------------------------------------------- protocol
    def bind(self, plant: Plant) -> None:
        self._unsub_all()
        self._reset_gateway()
        self.devices = {r: RoomDevice(r, plant) for r in ROOM_IDS}
        for r, d in self.devices.items():
            self._sub(self.topic(r, "cmd"), _DeviceEndpoint(self, d))
        self._sub(f"{self.prefix}/+/telemetry", self._on_telemetry)
        self._sub(f"{self.prefix}/+/ack", self._on_ack)

    def publish(self, now_min: int) -> None:
        """Every device publishes one reading stamped with the simulated time."""
        self._tick(now_min)
        self._now = now_min
        expect: dict[str, int] = {}
        for r, d in self.devices.items():
            tel = d.read(now_min)
            expect[r] = tel.seq
            self.bus.publish(self.topic(r, "telemetry"),
                             {"msg_id": f"{r}/telemetry/{tel.seq}", "run": self.run_id, **tel.to_dict()})
        self.bus.pump(
            until=lambda: all(self._last_seq.get(r, 0) >= s for r, s in expect.items()),
            timeout=self.telemetry_timeout_s,
        )

    def latest(self) -> dict[str, Telemetry]:
        return dict(self._latest)

    def command(self, cmds: list[Command], now_min: int) -> dict[str, Ack | None]:
        self._tick(now_min)
        self._now = now_min
        wanted = {c.cmd_id: c for c in cmds}
        for attempt in range(1 + self.max_retries):
            todo = [c for cid, c in wanted.items() if cid not in self._acks]
            if not todo:
                break
            if attempt > 0:
                self.stats["cmd_retries"] += len(todo)
            for c in todo:
                self._cmd_seq += 1
                self._sent.add(c.cmd_id)
                payload = {**c.to_dict(), "seq": self._cmd_seq, "attempt": attempt,
                           "msg_id": f"{c.room}/cmd/{self._cmd_seq}", "run": self.run_id}
                self.stats["cmds_sent"] += 1
                self.bus.publish(self.topic(c.room, "cmd"), payload)
            self.bus.pump(until=lambda: all(cid in self._acks for cid in wanted), timeout=self.ack_timeout_s)
        out: dict[str, Ack | None] = {}
        for c in cmds:
            a = self._acks.get(c.cmd_id)
            if a is None:
                self.stats["missing_acks"] += 1
            out[c.room] = a
        return out

    def close(self) -> None:
        self._unsub_all()
        if self.owns_bus:
            self.bus.close()               # e.g. stops the MQTT network thread and disconnects

    # ------------------------------------------------------- simulated time
    def _tick(self, now_min: int) -> None:
        """Advance the in-process bus clock so delayed messages arrive intervals later."""
        adv = getattr(self.bus, "advance", None)
        if self._last_tick_min is None:
            self._last_tick_min = now_min
            return
        if adv is not None and now_min > self._last_tick_min:
            adv((now_min - self._last_tick_min) // STEP_MIN or 1)
        self._last_tick_min = max(self._last_tick_min, now_min)

    # ---------------------------------------------------------- device side
    def _foreign(self, payload: dict[str, Any]) -> bool:
        if payload.get("run") != self.run_id:
            self.stats["foreign_msgs"] += 1
            return True
        return False

    def _on_cmd(self, device: RoomDevice, payload: dict[str, Any]) -> None:
        if self._foreign(payload):
            return
        try:
            cmd = Command.from_dict(payload)
        except (KeyError, TypeError):
            return
        ack = device.handle(cmd, self._now)
        if ack.duplicate:
            self.stats["cmd_replays_at_device"] += 1
        seq = self._ack_seq.get(device.room, 0) + 1
        self._ack_seq[device.room] = seq
        self.bus.publish(self.topic(device.room, "ack"),
                         {"msg_id": f"{device.room}/ack/{seq}", "seq": seq, "run": self.run_id, **ack.to_dict()})

    # --------------------------------------------------------- gateway side
    def _on_telemetry(self, topic: str, payload: dict[str, Any]) -> None:
        if self._foreign(payload):
            return
        try:
            tel = Telemetry.from_dict(payload)
        except (KeyError, TypeError):
            return
        last = self._last_seq.get(tel.room, 0)
        if tel.seq == last:
            self.stats["telemetry_duplicates"] += 1
            return
        if tel.seq < last:
            self.stats["telemetry_out_of_order"] += 1
            return
        self.stats["telemetry_received"] += 1
        self._last_seq[tel.room] = tel.seq
        self._latest[tel.room] = tel

    def _on_ack(self, topic: str, payload: dict[str, Any]) -> None:
        if self._foreign(payload):
            return
        try:
            ack = Ack.from_dict(payload)
        except (KeyError, TypeError):
            return
        if ack.cmd_id not in self._sent:
            self.stats["unknown_acks"] += 1
            return
        if ack.cmd_id in self._acks:
            self.stats["ack_duplicates"] += 1
            return
        self.stats["acks_received"] += 1
        self._acks[ack.cmd_id] = ack


def make_device_layer(kind: str = "direct", **kw: Any):
    """``direct`` | ``bus`` (in-process bus) | ``mqtt`` (falls back to the in-process bus).

    MQTT options: host, port, connect_timeout_s, prefix. When the broker is unreachable or
    paho-mqtt is missing, the returned layer has ``fallback=True`` and ``fallback_reason``.
    """
    if kind == "direct":
        return DirectDeviceLayer()
    layer_kw = {k: kw.pop(k) for k in ("prefix", "max_retries", "ack_timeout_s", "telemetry_timeout_s") if k in kw}
    if kind in ("bus", "inprocess"):
        return BusDeviceLayer(InProcessBus(), owns_bus=True, **layer_kw)
    if kind == "mqtt":
        from .mqtt_bus import MqttBus, MqttUnavailable

        try:
            return BusDeviceLayer(MqttBus(**kw), owns_bus=True, **layer_kw)
        except MqttUnavailable as e:
            log.warning("MQTT unavailable (%s); falling back to the in-process bus", e)
            layer = BusDeviceLayer(InProcessBus(), owns_bus=True, **layer_kw)
            layer.fallback = True
            layer.fallback_reason = str(e)
            return layer
    raise ValueError(f"unknown device layer kind {kind!r}")
