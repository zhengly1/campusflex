"""Acceptance tests for the IoT device layer (Table 11, owner: Member 2).

Duplicate, out-of-order and dropped messages, a missing acknowledgement, a stale reading
(older than two intervals), idempotent command replay, and a full episode through the
bus giving the same metrics as the in-process run. MQTT tests are skipped unless
paho-mqtt is installed and a broker listens on localhost:1883.
"""
from __future__ import annotations

import socket
import uuid

import numpy as np
import pytest

from campusflex.config import ROOM_INDEX, PlannerConfig
from campusflex.devices import Command, Plant
from campusflex.env import CampusEnv
from campusflex.episode import run_episode
from campusflex.iot import BusDeviceLayer, InProcessBus, make_device_layer, topic_matches
from campusflex.plans import mask_action
from campusflex.policies import make_policy
from campusflex.scenarios import make_scenario

T0 = 18 * 60


def _layer() -> tuple[BusDeviceLayer, InProcessBus, Plant]:
    bus = InProcessBus()
    layer = BusDeviceLayer(bus)
    plant = Plant.initial()
    layer.bind(plant)
    return layer, bus, plant


def _broker_up(host: str = "localhost", port: int = 1883) -> bool:
    try:
        import paho.mqtt.client  # noqa: F401
    except ImportError:
        return False
    try:
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except OSError:
        return False


# ----------------------------------------------------------------- bus basics
def test_topic_wildcards():
    assert topic_matches("campus/+/telemetry", "campus/A101/telemetry")
    assert not topic_matches("campus/+/telemetry", "campus/A101/ack")
    assert topic_matches("campus/#", "campus/A101/cmd")
    assert not topic_matches("campus/A101/cmd", "campus/A102/cmd")
    assert not topic_matches("campus/+", "campus/A101/cmd")


def test_telemetry_reaches_gateway_with_seq_and_timestamp():
    layer, _, plant = _layer()
    layer.publish(T0)
    tel = layer.latest()
    assert set(tel) == set(ROOM_INDEX)
    assert all(t.seq == 1 and t.ts_min == T0 for t in tel.values())
    assert tel["B201"].temp_c == pytest.approx(float(plant.T[ROOM_INDEX["B201"]]))


# --------------------------------------------------------- duplicate messages
def test_duplicate_telemetry_is_ignored():
    layer, bus, _ = _layer()
    bus.duplicate_next("campus/+/telemetry", n=6)
    layer.publish(T0)
    assert layer.stats["telemetry_duplicates"] == 6
    assert layer.stats["telemetry_received"] == 6
    assert all(t.seq == 1 for t in layer.latest().values())


def test_duplicate_command_is_applied_once():
    layer, bus, plant = _layer()
    bus.duplicate_next("campus/A101/cmd", n=1)
    acks = layer.command([Command("k1:A101", "A101", True, 26.0, T0)], T0)
    a = acks["A101"]
    assert a is not None and a.ok and not a.duplicate        # the first ack wins
    assert layer.stats["cmd_replays_at_device"] == 1           # the copy was recognised by the device
    assert layer.stats["ack_duplicates"] == 1
    assert len(layer.devices["A101"]._acks) == 1
    assert bool(plant.open[ROOM_INDEX["A101"]])


# ------------------------------------------------------- out-of-order messages
def test_out_of_order_telemetry_keeps_latest():
    layer, bus, plant = _layer()
    bus.hold_next("campus/A101/telemetry", n=2)
    layer.publish(T0)
    plant.T[ROOM_INDEX["A101"]] = 25.0
    layer.publish(T0 + 10)
    assert "A101" not in layer.latest()
    assert bus.release_held(reverse=True) == 2                  # seq 2 arrives before seq 1
    t = layer.latest()["A101"]
    assert (t.seq, t.ts_min, t.temp_c) == (2, T0 + 10, 25.0)
    assert layer.stats["telemetry_out_of_order"] == 1


# ------------------------------------------------------------ dropped messages
def test_dropped_command_is_retried_with_same_id():
    layer, bus, plant = _layer()
    bus.drop_next("campus/A101/cmd", n=1)
    acks = layer.command([Command("k2:A101", "A101", True, 26.0, T0)], T0)
    assert acks["A101"] is not None and acks["A101"].cmd_id == "k2:A101"
    assert layer.stats["cmd_retries"] == 1
    assert bus.stats.dropped == 1
    assert bool(plant.open[ROOM_INDEX["A101"]])


def test_lost_ack_retry_is_not_reapplied():
    layer, bus, _ = _layer()
    bus.drop_next("campus/A101/ack", n=1)
    acks = layer.command([Command("k3:A101", "A101", True, 26.0, T0)], T0)
    a = acks["A101"]
    assert a is not None and a.duplicate                       # re-ack of the stored result
    assert layer.stats["cmd_replays_at_device"] == 1
    assert len(layer.devices["A101"]._acks) == 1


def test_dropped_telemetry_keeps_previous_reading():
    layer, bus, _ = _layer()
    layer.publish(T0)
    bus.drop_next("campus/B202/telemetry", n=1)
    layer.publish(T0 + 10)
    tel = layer.latest()
    assert tel["B202"].ts_min == T0 and tel["A101"].ts_min == T0 + 10


# ---------------------------------------------------- missing acknowledgement
@pytest.mark.parametrize("lost", ["ack", "cmd"])
def test_missing_ack_leaves_room_and_ledger_unchanged(lost):
    cfg = PlannerConfig()
    bus = InProcessBus()
    env = CampusEnv(make_scenario("steady", 0), cfg, BusDeviceLayer(bus))
    snap = env.get_state()
    action = mask_action(frozenset({"A101", "B201"}), snap.health, cfg)
    bus.drop_next(f"campus/B201/{lost}", n=10)
    res = env.apply_plan(action, snap.version, "missing-ack-1")
    assert res.executed
    assert res.outcomes["B201"] == "no_ack" and res.outcomes["A101"] == "ack"
    assert env.counters["no_ack"] == 1
    after = env.get_state()
    b, a = ROOM_INDEX["B201"], ROOM_INDEX["A101"]
    assert not after.open[b] and after.counts[b] == 0          # nothing assigned to the unconfirmed room
    assert after.open[a] and after.counts[a] == min(60, snap.demand)
    if lost == "cmd":
        assert not env.plant.open[b]                            # device never received it


# ------------------------------------------------------------- stale reading
def test_stale_reading_flagged_after_two_intervals():
    cfg = PlannerConfig()
    bus = InProcessBus()
    env = CampusEnv(make_scenario("steady", 0), cfg, BusDeviceLayer(bus))
    i = ROOM_INDEX["A103"]
    assert not env.get_state().stale.any()
    bus.drop_next("campus/A103/telemetry", n=3)
    snap = None
    for k in range(1, 4):
        snap = env.advance_time()
        if k <= 2:                                              # 10 and 20 minutes old: not yet stale
            assert not snap.stale[i]
    assert snap.stale[i]                                        # 30 minutes old
    assert snap.stale.sum() == 1
    assert snap.temps[i] == pytest.approx(env.device.latest()["A103"].temp_c)


def test_delayed_telemetry_arrives_next_interval():
    layer, bus, _ = _layer()
    layer.publish(T0)
    bus.delay_next("campus/A102/telemetry", n=1, ticks=2)
    layer.publish(T0 + 10)                                      # seq 2 delayed by two intervals
    assert layer.latest()["A102"].ts_min == T0
    layer.publish(T0 + 20)                                      # seq 3 arrives first
    assert layer.latest()["A102"].seq == 3
    layer.publish(T0 + 30)                                      # late seq 2 is discarded as out of order
    assert layer.latest()["A102"].ts_min == T0 + 30
    assert layer.stats["telemetry_out_of_order"] == 1


# --------------------------------------------------------- idempotent replay
def test_command_id_replay_is_idempotent():
    layer, _, plant = _layer()
    x = Command("X:A101", "A101", True, 26.0, T0)
    y = Command("Y:A101", "A101", False, None, T0 + 10)
    assert layer.command([x], T0)["A101"].ok
    assert layer.command([y], T0 + 10)["A101"].ok
    assert not plant.open[ROOM_INDEX["A101"]]
    r = layer.command([x], T0 + 20)["A101"]                     # replay of an old key
    assert r is not None and r.cmd_id == "X:A101"
    assert not plant.open[ROOM_INDEX["A101"]]                   # not re-applied


def test_env_duplicate_idempotency_key_executes_once():
    cfg = PlannerConfig()
    env = CampusEnv(make_scenario("steady", 0), cfg, BusDeviceLayer(InProcessBus()))
    snap = env.get_state()
    a = mask_action(frozenset({"A101", "A102"}), snap.health, cfg)
    r1 = env.apply_plan(a, snap.version, "dup-key")
    counts = env.ledger.counts().copy()
    r2 = env.apply_plan(a, snap.version, "dup-key")
    assert r1.executed and not r2.executed and r2.duplicate
    assert np.array_equal(env.ledger.counts(), counts)


def test_ac_command_to_offline_unit_is_refused_by_device():
    layer, _, plant = _layer()
    plant.health[ROOM_INDEX["B201"]] = 0.0
    a = layer.command([Command("k4:B201", "B201", True, 26.0, T0)], T0)["B201"]
    assert a is not None and not a.ok and a.reason == "ac_offline" and a.setpoint is None
    assert np.isnan(plant.setpoint[ROOM_INDEX["B201"]])


# ------------------------------------------------ full episode: bus == direct
@pytest.mark.parametrize("policy", ["planner", "threshold"])
def test_full_episode_through_bus_matches_direct(policy):
    cfg = PlannerConfig()
    sc = make_scenario("fault", 0)
    direct = run_episode(sc, make_policy(policy, cfg), cfg)
    layer = BusDeviceLayer(InProcessBus())
    via_bus = run_episode(sc, make_policy(policy, cfg), cfg, device_layer=layer)
    assert via_bus.metrics == direct.metrics
    assert [s["counts"] for s in via_bus.steps] == [s["counts"] for s in direct.steps]
    assert layer.stats["missing_acks"] == 0 and layer.stats["acks_received"] == 6 * 24


# ------------------------------------------------------------ factory / MQTT
def test_factory_direct_and_bus():
    assert make_device_layer("direct").name == "direct"
    lay = make_device_layer("bus")
    assert isinstance(lay, BusDeviceLayer) and lay.name == "bus:inprocess" and not lay.fallback
    lay.close()
    assert lay.bus.closed                  # a bus the factory created is closed with its layer
    shared = InProcessBus()
    BusDeviceLayer(shared).close()
    assert not getattr(shared, "closed", False)   # a caller's bus is left open


def test_mqtt_unreachable_falls_back_and_reports_it():
    lay = make_device_layer("mqtt", host="127.0.0.1", port=1, connect_timeout_s=0.5)
    assert isinstance(lay, BusDeviceLayer)
    assert lay.fallback and lay.fallback_reason
    assert isinstance(lay.bus, InProcessBus)


_NEEDS_BROKER = pytest.mark.skipif(not _broker_up(), reason="paho-mqtt missing or no MQTT broker on localhost:1883")


@_NEEDS_BROKER
@pytest.mark.parametrize("policy", ["planner", "threshold"])
def test_full_episode_through_mqtt_broker_matches_direct(policy):
    cfg = PlannerConfig()
    sc = make_scenario("fault", 0)
    direct = run_episode(sc, make_policy(policy, cfg), cfg)
    layer = make_device_layer("mqtt", prefix=f"campus-test-{uuid.uuid4().hex[:6]}")
    assert not layer.fallback, layer.fallback_reason
    try:
        via_mqtt = run_episode(sc, make_policy(policy, cfg), cfg, device_layer=layer)
    finally:
        layer.close()
    assert via_mqtt.metrics == direct.metrics
    assert [s["counts"] for s in via_mqtt.steps] == [s["counts"] for s in direct.steps]
    assert layer.stats["missing_acks"] == 0 and layer.stats["acks_received"] == 6 * 24
    assert layer.stats["foreign_msgs"] == 0


@_NEEDS_BROKER
def test_runs_sharing_a_real_broker_do_not_control_each_other():
    """Two runs on one broker with the same topic prefix: each ignores the other's messages."""
    cfg = PlannerConfig()
    prefix = f"campus-test-{uuid.uuid4().hex[:6]}"
    l1, l2 = make_device_layer("mqtt", prefix=prefix), make_device_layer("mqtt", prefix=prefix)
    assert not (l1.fallback or l2.fallback)
    try:
        e1 = CampusEnv(make_scenario("steady", 0), cfg, l1)
        e2 = CampusEnv(make_scenario("steady", 1), cfg, l2)
        s = e1.get_state()
        assert e1.apply_plan(mask_action(frozenset({"A103"}), s.health, cfg), s.version, "k").executed
        e2.advance_time()                      # e2 now drains e1's command, ack and telemetry
        assert e1.plant.open[ROOM_INDEX["A103"]] and not e2.plant.open.any()
        assert l2.stats["foreign_msgs"] > 0 and l2.stats["unknown_acks"] == 0
    finally:
        l1.close()
        l2.close()


# ------------------------------------------- belief vs plant (drift), partial acks
def test_lost_close_ack_is_detected_as_drift_and_reported():
    """The device closes B202 but the ack is lost: the next reading corrects the belief,
    the occupied-closed-room violation is reported, and the students are relocated."""
    cfg = PlannerConfig()
    bus = InProcessBus()
    env = CampusEnv(make_scenario("steady", 0), cfg, BusDeviceLayer(bus))
    b = ROOM_INDEX["B202"]
    s = env.get_state()
    for t in range(4):                                         # B202 + A101 open long enough to close
        assert env.apply_plan(mask_action(frozenset({"A101", "B202"}), s.health, cfg), s.version, f"o{t}").executed
        s = env.advance_time()
    assert s.counts[b] > 0
    bus.drop_next("campus/B202/ack", n=10)
    res = env.apply_plan(mask_action(frozenset({"A101", "A102", "B201"}), s.health, cfg), s.version, "close")
    assert res.outcomes["B202"] == "no_ack"
    assert not env.plant.open[b] and env.believed_open[b]      # device closed it; belief unchanged
    s2 = env.advance_time()
    assert env.step_logs[-1]["violations"] >= 1                # students sat in a closed room: reported
    assert env.counters["drift"] >= 1 and not s2.open[b]       # belief corrected from telemetry
    assert env.apply_plan(mask_action(frozenset({"A101", "A102", "B201"}), s2.health, cfg), s2.version, "next").executed
    assert env.ledger.counts()[b] == 0                         # occupants relocated, not stranded


def test_partial_ack_defers_close_make_before_break():
    cfg = PlannerConfig()
    bus = InProcessBus()
    env = CampusEnv(make_scenario("steady", 0), cfg, BusDeviceLayer(bus))
    s = env.get_state()
    for t in range(4):
        assert env.apply_plan(mask_action(frozenset({"B201"}), s.health, cfg), s.version, f"o{t}").executed
        s = env.advance_time()
    n_before = int(s.counts.sum())
    bus.drop_next("campus/B202/ack", n=10)                      # the new room never confirms
    res = env.apply_plan(mask_action(frozenset({"B202", "A101"}), s.health, cfg), s.version, "swap")
    assert res.outcomes["B201"] == "deferred" and env.plant.open[ROOM_INDEX["B201"]]
    assert env.counters["deferred_close"] == 1
    env.advance_time()
    assert env.step_logs[-1]["LS"] == 0 and env.step_logs[-1]["served"] >= n_before


def test_ac_offline_ack_marks_unit_offline_even_if_sensor_is_silent():
    cfg = PlannerConfig()
    bus = InProcessBus()
    env = CampusEnv(make_scenario("steady", 0), cfg, BusDeviceLayer(bus))
    i = ROOM_INDEX["B201"]
    s = env.get_state()
    assert env.apply_plan(mask_action(frozenset({"B201"}), s.health, cfg), s.version, "a").executed
    env.plant.health[i] = 0.0                                   # unit fails, sensor goes silent
    bus.drop_next("campus/B201/telemetry", n=50)
    env.sc.health[1:, i] = 0.0
    s = env.advance_time()
    assert s.health[i] == 1.0                                   # last reading still says healthy
    a = mask_action(frozenset({"B201"}), s.health, cfg)
    res = env.apply_plan(a, s.version, "b")
    assert res.outcomes["B201"] == "ac_offline"
    env.advance_time()
    assert env.get_state().health[i] == 0.0


def test_device_refuses_out_of_order_command():
    layer, bus, plant = _layer()
    i = ROOM_INDEX["A103"]
    bus.delay_next("campus/A103/cmd", n=1, ticks=2)
    layer.command([Command("old:A103", "A103", True, 26.0, T0)], T0)        # delayed open from 18:00
    layer.command([Command("new:A103", "A103", False, None, T0 + 10)], T0 + 10)
    assert not plant.open[i]
    layer.publish(T0 + 30)                                                  # the delayed open arrives
    assert not plant.open[i]


def test_runs_sharing_a_bus_do_not_control_each_other():
    bus = InProcessBus()
    cfg = PlannerConfig()
    e1 = CampusEnv(make_scenario("steady", 0), cfg, BusDeviceLayer(bus))
    e2 = CampusEnv(make_scenario("steady", 1), cfg, BusDeviceLayer(bus))
    s = e1.get_state()
    assert e1.apply_plan(mask_action(frozenset({"A103"}), s.health, cfg), s.version, "k").executed
    assert e1.plant.open[ROOM_INDEX["A103"]] and not e2.plant.open.any()


def test_graph_snapshot_keeps_recheck_requests():
    from campusflex.agent.graph import CampusFlexGraph

    cfg = PlannerConfig()
    env = CampusEnv(make_scenario("steady", 0), cfg)
    seen = []
    g = CampusFlexGraph(env, make_policy("arm_A", cfg), cfg, run_id="g", force_minigraph=True,
                        on_step=lambda e, snap, d, c: (seen.append(list(snap.rechecks)),
                                                       e.request_recheck("A101") if snap.step == 2 else None))
    g.run()
    assert seen[3] == ["A101"]
