"""Smart device extension: per-device IoT dispatch, learned usage profiles, sensor-driven control.

This package sits on top of the core (which decides which rooms are used and who sits
where) and decides how every device in every room runs:

* ``devices`` / ``iot`` – six devices per room (AC with fan speed, dimmable light, fresh
  air, curtain, projector, sockets), validated idempotent commands over MQTT topics
  ``campus/<room>/<device>/cmd|ack`` and ``campus/<room>/sensors`` (IoT);
* ``profiles`` – what people in each room and hourly slot set by hand, learned and
  remembered across evenings, with slot personas (AI);
* ``sensors`` / ``dsa`` – noisy sensors with faults; Kalman filtering, spike/stuck/drift
  detection, online thermal-model identification (DSA);
* ``scheduler`` – static and rule baselines and the smart scheduler; ``run`` simulates
  evenings; ``campusflex.experiments.smart`` is the experiment.

The core experiments (E1–E6) and their frozen configuration do not use this package.
"""
from .devices import DEVICES, DeviceState, SmartAck, SmartCommand, SmartRoomFirmware
from .profiles import ProfileStore, slot_label, slot_of
from .scheduler import RuleScheduler, SmartScheduler, StaticScheduler

__all__ = [
    "DEVICES",
    "DeviceState",
    "ProfileStore",
    "RuleScheduler",
    "SmartAck",
    "SmartCommand",
    "SmartRoomFirmware",
    "SmartScheduler",
    "StaticScheduler",
    "slot_label",
    "slot_of",
]
