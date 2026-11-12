"""Smart device extension: device firmware and IoT transport, room physics, sensors and DSA,
usage profiles, schedulers, and a small end-to-end experiment."""
from __future__ import annotations

import json

import numpy as np
import pytest

from campusflex.config import ROOM_BY_ID
from campusflex.iot.bus import InProcessBus
from campusflex.smart.devices import (
    DeviceState,
    SmartCommand,
    SmartRoomFirmware,
    validate_value,
)
from campusflex.smart.dsa import RoomEstimator, score_detection
from campusflex.smart.iot import SmartIoT
from campusflex.smart.physics import E_MAX_LUX, SmartPlant, daylight_lux
from campusflex.smart.profiles import ProfileStore, narrate, slot_of
from campusflex.smart.scheduler import OFF, RoomView, SmartScheduler
from campusflex.smart.sensors import SensorFault, SensorSuite


def _cmd(device, value, cid="c1", seq=1, room="A101"):
    return SmartCommand(cid, room, device, value, 1100, seq=seq)


# ----------------------------------------------------------------- firmware / IoT
def test_firmware_validates_refuses_offline_ac_and_is_idempotent():
    st = DeviceState()
    health = {"h": 1.0}
    fw = SmartRoomFirmware("A101", st, health=lambda: health["h"])
    assert validate_value("light", {"level": 55}) and validate_value("ac", {"setpoint": 30.5, "fan": 2})
    assert not fw.handle(_cmd("light", {"level": 55})).ok
    ok = fw.handle(_cmd("ac", {"setpoint": 24.5, "fan": 2}, "c2", 1))
    assert ok.ok and st.ac_setpoint == 24.5 and st.ac_fan == 2
    replay = fw.handle(_cmd("ac", {"setpoint": 26.0, "fan": 1}, "c2", 1))
    assert replay.duplicate and st.ac_setpoint == 24.5                       # same key: not applied twice
    assert fw.handle(_cmd("ac", {"setpoint": 25.0, "fan": 1}, "c3", 1)).reason == "out_of_order"
    health["h"] = 0.0
    assert fw.handle(_cmd("ac", {"setpoint": 25.0, "fan": 1}, "c4", 2)).reason == "ac_offline"
    assert fw.handle(_cmd("ac", {"setpoint": None, "fan": 0}, "c5", 3)).ok   # switching off is always allowed
    assert not st.ac_on()


def test_iot_commands_survive_drops_and_duplicates_and_ignore_other_runs():
    bus = InProcessBus()
    plant = SmartPlant()
    iot = SmartIoT(bus, plant)
    bus.drop_next("campus/A101/light/cmd", 1)                      # first attempt lost → retried, same key
    ack = iot.command("A101", "light", {"level": 70}, 1100)
    assert ack.ok and plant.devices["A101"].light == 70 and iot.stats["retries"] == 1
    assert iot.applied["A101"].light == 70                         # gateway mirror follows the ack
    bus.duplicate_next("campus/B201/vent/cmd", 1)
    assert iot.command("B201", "vent", {"level": 2}, 1102).ok and iot.stats["acks"] == 2
    other = SmartIoT(bus, plant)                                   # a second run on the same bus
    other.command("A101", "light", {"level": 30}, 1104)
    assert iot.stats["foreign"] > 0 and iot.applied["A101"].light == 70
    iot.close()
    other.close()


# -------------------------------------------------------------------- physics
def test_room_physics_co2_light_fan_and_equipment_heat():
    def run(**dev):
        p = SmartPlant()
        for k, v in dev.items():
            setattr(p.devices["A101"], k, v)
        rec = None
        for m in range(1080, 1140, 2):
            rec = p.step(m, 29.0, {"A101": 60})["A101"]
        return rec
    stale, fresh = run(vent=0), run(vent=3)
    assert stale.co2_ppm > 1500 and fresh.co2_ppm < 1000
    lit = run(light=60, curtain=0)
    assert abs(lit.lux - 0.6 * E_MAX_LUX) < 1e-6
    weak, strong = run(ac_setpoint=24.0, ac_fan=1), run(ac_setpoint=24.0, ac_fan=3)
    assert strong.temp_c < weak.temp_c                              # fan speed limits cooling capacity
    assert run(projector=True, sockets=True).temp_c > run().temp_c   # equipment heat
    assert daylight_lux("B201", 1080) > daylight_lux("A101", 1080) > 0 == daylight_lux("A101", 1140)


# ------------------------------------------------------------- sensors / DSA
def _estimate_day(faults, minutes=range(1080, 1240, 2)):
    p = SmartPlant()
    s = p.devices["A101"]
    s.ac_setpoint, s.ac_fan, s.vent, s.light = 25.0, 2, 1, 60
    sensors = SensorSuite(3, faults, dropout=0.0)
    est = RoomEstimator("A101")
    for m in minutes:
        rec = p.step(m, 29.0, {"A101": 50})["A101"]
        est.update(sensors.read(rec), 29.0, s)
    return est


def test_dsa_flags_stuck_and_drift_and_raises_no_false_alarm():
    assert _estimate_day([]).flags == []
    stuck = _estimate_day([SensorFault("A101", "co2", "stuck", 1150)])
    assert [(f["sensor"], f["kind"]) for f in stuck.flags] == [("co2", "stuck")]
    drift = _estimate_day([SensorFault("A101", "temp", "drift", 1150, magnitude=0.05)])
    assert [(f["sensor"], f["kind"]) for f in drift.flags] == [("temp", "drift")]
    assert drift.kf["temp"].x < 25.6                                 # estimate stays with the model, not the drift
    spike = _estimate_day([SensorFault("A101", "lux", "spike", 1150, magnitude=400)])
    assert spike.flags == [] and abs(spike.kf["lux"].x - 0.6 * E_MAX_LUX) < 60
    f = SensorFault("A101", "co2", "stuck", 1150)
    sc = score_detection(stuck.flags, [f])
    assert sc["recall"] == 1.0 and sc["false_alarms"] == 0 and 0 <= sc["mean_delay_min"] <= 30


def test_thermal_model_is_identified_across_evenings():
    from campusflex.smart.run import day_plan, run_day

    models = {}
    for d in range(6):
        run_day(day_plan(d, ("steady", "surge", "heat")[d % 3], d), SmartScheduler(None, True), models=models)
    errs = [abs(m.identified()["C"] / ROOM_BY_ID[r].C - 1) for r, m in models.items() if m.identified()]
    assert len(errs) >= 3 and float(np.median(errs)) < 0.4


# ------------------------------------------------------------------ profiles
def test_profiles_learn_pool_decay_and_persist(tmp_path):
    st = ProfileStore()
    for _ in range(5):
        st.observe_adjustment("A101", 1, "ac", {"setpoint": 24.0, "fan": 3})
        st.observe_adjustment("A101", 1, "projector", {"on": True})
        st.observe_adjustment("A101", 1, "light", {"level": 50}, lux_after=280.0)
    rec = st.recommend("A101", 1)
    assert rec["source"]["setpoint"] == "room" and 24.0 <= rec["setpoint"] < 24.5 and rec["fan"] == 3
    assert rec["projector"] and abs(rec["lux"] - 280.0) < 60
    other = st.recommend("B203", 1)                                 # never seen: the slot's campus habit
    assert other["source"]["setpoint"] == "slot" and other["setpoint"] == 24.0
    assert st.recommend("A101", 3)["source"]["setpoint"] == "default"
    w0 = st.cells["A101|1|setpoint"].w
    st.end_day()
    assert st.cells["A101|1|setpoint"].w == pytest.approx(w0 * st.decay)
    st.save(tmp_path / "p.json")
    back = ProfileStore.load(tmp_path / "p.json")
    assert back.recommend("A101", 1) == st.recommend("A101", 1) and back.days == 1
    tags = st.personas()[1]["tags"]
    assert "lecture (projector on)" in tags and "prefers cool" in tags and "strong airflow" in tags
    assert slot_of(1080) == 0 and slot_of(1199) == 1 and slot_of(1319) == 3


def test_narration_template_and_llm_numbers_are_checked():
    from campusflex.llm.client import LLMResponse, ScriptedClient

    st = ProfileStore()
    st.observe_adjustment("A101", 1, "ac", {"setpoint": 24.0, "fan": 3})
    t = narrate(st)
    assert t["source"] == "template" and len(t["lines"]) == 4 and "24.0 °C" in t["lines"][1]
    bogus = ScriptedClient(lambda role, s, u: LLMResponse(role, {"lines": ["Lectures run at 24.0 °C with 73 people."]}))
    out = narrate(st, bogus)
    assert out["source"] == "llm" and "73" in out["flagged_numbers"] and "[unverified]" in out["lines"][0]


# ----------------------------------------------------------------- scheduler
def _view(**kw):
    base = dict(room="B201", minute=1090, slot=0, open_plan=True, n_plan=40, n_next=40, minutes_to_arrival=None,
                state=DeviceState(ac_setpoint=25.0, ac_fan=2, vent=1, curtain=60), t_out=29.0)
    base.update(kw)
    return RoomView(**base)


def test_smart_scheduler_rules():
    s = SmartScheduler(None, use_dsa=False)
    empty = s.decide(_view(n_plan=0, n_next=0, open_plan=False))
    assert {k: empty[k] for k in OFF} == OFF and empty["curtain"] == {"open_pct": 0}   # sun kept out of an empty B room
    dsa = SmartScheduler(None, use_dsa=True)
    from campusflex.smart.dsa import Estimate

    def est(**kw):
        e = dict(room="B201", minute=1090, temp_c=25.1, co2_ppm=700.0, lux=400.0, daylight_lux=0.0, occupancy=40,
                 verdicts={}, faulty={})
        e.update(kw)
        return Estimate(**e)
    dark, bright = (dsa.decide(_view(est=est(daylight_lux=d)))["light"]["level"] for d in (0.0, 400.0))
    assert bright < dark                                            # daylight harvesting
    stuffy = dsa.decide(_view(n_plan=70, n_next=70, est=est(co2_ppm=1300.0)))["vent"]["level"]
    assert stuffy == 3
    manual = dsa.decide(_view(est=est(temp_c=23.7), user_recent={"ac": {"setpoint": 23.5, "fan": 1}}))
    assert manual["ac"] == {"setpoint": 23.5, "fan": 1}             # a manual choice is not undone in its slot
    hot = dsa.decide(_view(est=est(temp_c=26.5), state=DeviceState(ac_setpoint=25.0, ac_fan=2, vent=1)))
    assert hot["ac"]["fan"] == 2                                    # unknown tolerance: no escalation beyond the default
    known = dsa.decide(_view(est=est(temp_c=24.6), user_recent={"ac": {"setpoint": 23.5, "fan": 1}},
                             state=DeviceState(ac_setpoint=23.5, ac_fan=1, vent=1)))
    assert known["ac"]["fan"] == 2                                  # one level above a fan people accepted
    glare = dsa.decide(_view(minute=1082, est=est(daylight_lux=450.0)))
    st = ProfileStore()
    for _ in range(3):
        st.observe_adjustment("B201", 0, "projector", {"on": True})
    lecture = SmartScheduler(st, True).decide(_view(minute=1082, est=est(daylight_lux=600.0)))   # 30 % open → 180 lux on the screen
    assert lecture["projector"]["on"] and lecture["curtain"] == {"open_pct": 0} and glare["projector"] == {"on": False}


# ------------------------------------------------------------ end to end
def test_small_experiment_learns_and_writes_its_memory(tmp_path):
    from campusflex.experiments import smart as S

    res = S.run(train_days=4, eval_days=2, workers=4, out_dir=tmp_path)
    assert set(res["eval"]) == set(S.ARMS)
    assert res["eval"]["smart"]["adjustments_per_evening"] < res["eval"]["rule"]["adjustments_per_evening"]
    assert res["eval"]["smart"]["discomfort_pct"]["co2"] <= res["eval"]["static"]["discomfort_pct"]["co2"]
    assert res["anomaly_detection"]["faults"] == 4 - sum(f["kind"] == "spike" for f in res["faults_injected"])
    saved = json.loads((tmp_path / "profiles.json").read_text())
    assert saved["days"] == 4 and saved["n_adjustments"] > 0
    assert (tmp_path / "personas.md").read_text().startswith("# Learned usage personas")
    assert len(res["profile_grid"]) == 6 * 4 and {"room", "slot", "setpoint", "source"} <= set(res["profile_grid"][0])
    evg = json.loads((tmp_path / "evening.json").read_text())       # one evaluation evening, traced in every arm
    assert set(evg["arms"]) == set(S.ARMS) and evg["meta"]["eval_day_index"] == S.TRACE_DAY
    for x in evg["arms"].values():
        assert x["metrics"]["adjustments"] == sum(e["by"] == "user" and e["ok"] for t in x["ticks"] for e in t["events"])
    warm = S.run_arm(("smart", [], [], saved, 0))                   # a saved memory is a valid starting point
    assert warm["profiles"]["days"] == 4


def test_trace_records_the_evening_without_changing_it():
    from campusflex.experiments.smart import eval_faults
    from campusflex.smart.run import day_plan, run_day

    plan = day_plan(20, "steady", 100)
    plan.faults = eval_faults(plan, 0)                              # A101: temperature stuck, CO2 drift
    plain = run_day(plan, SmartScheduler(ProfileStore(), use_dsa=True), models={})
    ticks: list = []
    traced = run_day(plan, SmartScheduler(ProfileStore(), use_dsa=True), models={}, trace=ticks)
    assert traced == plain and len(ticks) == 120
    events = [e for t in ticks for e in t["events"]]
    assert sum(e["by"] == "user" and e["ok"] for e in events) == plain["adjustments"]
    assert sum(e["by"] == "scheduler" for e in events) == plain["scheduler_commands"]
    assert [f for t in ticks for f in t["flags"]] == plain["flags"]
    stuck = next(t for t in ticks if t["fault_active"][0].get("temp") == "stuck")
    assert stuck["clock"] == "19:00" and ticks[-1]["suspect"][0].get("temp") == "stuck"
    assert all(len(t[f]) == 6 for t in ticks for f in ("temp_true", "temp_sensor", "temp_est", "setpoint", "kw"))
