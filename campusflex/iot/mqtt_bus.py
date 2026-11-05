"""MQTT transport (local Mosquitto broker, paho-mqtt client, QoS 1, JSON payloads).

paho's network loop runs in a background thread; received messages are queued and
dispatched on the caller's thread inside ``pump`` so that handlers may publish (device
acks) without blocking the network loop. ``pump(until, timeout)`` returns as soon as
``until()`` is true, which gives the simulated-time device layer synchronous semantics.
"""
from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from typing import Any, Callable

from .bus import Handler, topic_matches


class MqttUnavailable(RuntimeError):
    """paho-mqtt is not installed or no broker is reachable."""


class MqttBus:
    name = "mqtt"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 1883,
        client_id: str | None = None,
        keepalive: int = 30,
        connect_timeout_s: float = 2.0,
        op_timeout_s: float = 2.0,
        qos: int = 1,
    ):
        try:
            import paho.mqtt.client as mqtt
        except ImportError as e:  # pragma: no cover - depends on environment
            raise MqttUnavailable("paho-mqtt is not installed (pip install paho-mqtt)") from e
        self._mqtt = mqtt
        self.host, self.port, self.qos = host, port, qos
        self.op_timeout_s = op_timeout_s
        self._q: "queue.Queue[tuple[str, bytes]]" = queue.Queue()
        self._subs: list[tuple[str, Handler]] = []
        self._connected = threading.Event()
        self._connect_rc: Any = None
        self._suback = threading.Condition()
        self._acked_mids: set[int] = set()
        self.client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id or f"campusflex-{uuid.uuid4().hex[:10]}",
            clean_session=True,
        )
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_subscribe = self._on_subscribe
        self.client.on_unsubscribe = self._on_subscribe
        try:
            self.client.connect(host, port, keepalive)
        except (OSError, ConnectionError) as e:
            raise MqttUnavailable(f"no MQTT broker reachable at {host}:{port}: {e}") from e
        self.client.loop_start()
        if not self._connected.wait(connect_timeout_s):
            self.client.loop_stop()
            raise MqttUnavailable(f"MQTT broker at {host}:{port} did not accept the connection (rc={self._connect_rc})")

    # ------------------------------------------------------------ callbacks (network thread)
    def _on_connect(self, client, userdata, flags, reason_code, properties=None) -> None:
        self._connect_rc = reason_code
        failed = getattr(reason_code, "is_failure", None)
        if failed is None:
            failed = reason_code != 0
        if not failed:
            self._connected.set()

    def _on_message(self, client, userdata, msg) -> None:
        self._q.put((msg.topic, msg.payload))

    def _on_subscribe(self, client, userdata, mid, *args) -> None:
        with self._suback:
            self._acked_mids.add(mid)
            self._suback.notify_all()

    def _wait_mid(self, mid: int) -> None:
        deadline = time.monotonic() + self.op_timeout_s
        with self._suback:
            while mid not in self._acked_mids:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError(f"MQTT (un)subscribe mid={mid} not acknowledged")
                self._suback.wait(left)
            self._acked_mids.discard(mid)

    # --------------------------------------------------------------------- Bus API
    def subscribe(self, pattern: str, handler: Handler) -> None:
        first = all(p != pattern for p, _ in self._subs)
        self._subs.append((pattern, handler))
        if first:
            rc, mid = self.client.subscribe(pattern, qos=self.qos)
            if rc != self._mqtt.MQTT_ERR_SUCCESS:
                raise MqttUnavailable(f"subscribe {pattern!r} failed rc={rc}")
            self._wait_mid(mid)

    def unsubscribe(self, pattern: str, handler: Handler) -> None:
        self._subs = [(p, h) for p, h in self._subs if not (p == pattern and h == handler)]
        if all(p != pattern for p, _ in self._subs):
            rc, mid = self.client.unsubscribe(pattern)
            if rc == self._mqtt.MQTT_ERR_SUCCESS:
                try:
                    self._wait_mid(mid)
                except TimeoutError:
                    pass

    def publish(self, topic: str, payload: dict[str, Any]) -> None:
        info = self.client.publish(topic, json.dumps(payload), qos=self.qos)
        try:
            info.wait_for_publish(timeout=self.op_timeout_s)
        except (RuntimeError, ValueError) as e:
            raise MqttUnavailable(f"publish to {topic!r} failed: {e}") from e

    def pump(self, until: Callable[[], bool] | None = None, timeout: float = 0.0) -> None:
        """Dispatch queued messages on this thread until ``until()`` holds or ``timeout`` expires.

        With ``until=None`` the queue is drained, waiting at most ``timeout`` for stragglers.
        """
        deadline = time.monotonic() + timeout
        while True:
            if until is not None and until():
                return
            left = deadline - time.monotonic()
            try:
                topic, raw = self._q.get(timeout=max(0.0, left)) if left > 0 else self._q.get_nowait()
            except queue.Empty:
                return
            try:
                payload = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue      # malformed message on our topics: ignore
            for pattern, handler in list(self._subs):
                if topic_matches(pattern, topic):
                    handler(topic, payload)

    def close(self) -> None:
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:  # pragma: no cover
            pass
        self._subs.clear()
