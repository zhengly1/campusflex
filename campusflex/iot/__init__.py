"""IoT device layer: MQTT transport with an in-process fallback (Section 3.7, Table 11)."""
from .bus import Bus, InProcessBus, topic_matches
from .layer import BusDeviceLayer, make_device_layer
from .mqtt_bus import MqttBus, MqttUnavailable

__all__ = [
    "Bus",
    "BusDeviceLayer",
    "InProcessBus",
    "MqttBus",
    "MqttUnavailable",
    "make_device_layer",
    "topic_matches",
]
