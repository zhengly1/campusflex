"""The 13 invariant and interface tests of the numerical core (Sections 3.1, 4.1).

They are rerun against every change to the core.
"""
from __future__ import annotations


import numpy as np
import pytest

from campusflex.config import DT_H, N_ROOMS, Q_OCC_KW, ROOM_IDS, ROOMS, TOTAL_SEATS, PlannerConfig
from campusflex.devices import DirectDeviceLayer
from campusflex.env import CampusEnv
from campusflex.episode import run_episode
from campusflex.metrics import episode_metrics
from campusflex.model import NOMINAL, SEATS, assign, deviation, power_kw, step_temps, thermostat_u
from campusflex.plans import StepAction, mask_action
from campusflex.policies import make_policy
from campusflex.scenarios import make_scenario
from campusflex.simulate import RolloutResult, Score, score_of

ALL_OPEN = np.ones(N_ROOMS, dtype=bool)
NONE_OPEN = np.zeros(N_ROOMS, dtype=bool)
ORDER = list(range(N_ROOMS))
T26 = np.full(N_ROOMS, 26.0)


def test_01_capacity_and_demand_accounting_340_request_300_seats():
    r = assign(np.zeros(N_ROOMS, int), NONE_OPEN, ALL_OPEN, SEATS, 340, T26, ORDER)
    assert r.counts.sum() == TOTAL_SEATS == 300
    assert r.unserved == 40
    assert np.all(r.counts <= SEATS)


def test_02_existing_occupants_preserved_under_arrivals():
    before = np.array([30, 10, 0, 20, 0, 0])
    r = assign(before, ALL_OPEN, ALL_OPEN, SEATS, 120, T26, ORDER)
    assert r.moved == 0
    assert np.all(r.counts >= before)
    assert r.counts.sum() == 120


def test_03_departures_are_not_moves():
    before = np.array([60, 50, 40, 0, 0, 0])
    r = assign(before, ALL_OPEN, ALL_OPEN, SEATS, 90, T26, ORDER)
    assert r.moved == 0 and r.counts.sum() == 90
    # in the environment ledger as well
    cfg = PlannerConfig()
    sc = make_scenario("surge", 0)
    res = run_episode(sc, make_policy("fixed", cfg), cfg)
    assert res.metrics["moves"] == 0          # fixed opening never relocates, even when demand falls at 21:00


def test_04_relocation_after_room_closes():
    before = np.array([0, 0, 0, 40, 0, 0])
    new_open = ALL_OPEN.copy()
    new_open[3] = False
    r = assign(before, ALL_OPEN, new_open, SEATS, 40, T26, ORDER)
    assert r.counts[3] == 0 and r.counts.sum() == 40
    assert r.moved == 40 and r.forced == 40 and r.switches == 1


def test_05_zero_cooling_from_offline_units():
    health = np.ones(N_ROOMS)
    health[3] = 0.0
    u = thermostat_u(np.full(N_ROOMS, 29.0), 31.0, np.full(N_ROOMS, 30), np.full(N_ROOMS, 24.0), health)
    assert u[3] == 0.0 and np.all(u <= health)
    cfg = PlannerConfig()
    sc = make_scenario("fault", 0)
    env = CampusEnv(sc, cfg)
    while env.step < 12:
        s = env.get_state()
        env.apply_plan(mask_action(frozenset(ROOM_IDS), s.health, cfg), s.version, f"k{env.step}")
        env.advance_time()
    s = env.get_state()
    a = StepAction(open=frozenset(ROOM_IDS), setpoints=(("B201", 26.0),))
    res = env.apply_plan(a, s.version, "offline")
    assert not res.executed and res.rejected[0]["code"] == "offline_command"


def test_06_thermal_step_matches_equation_2():
    T = np.array([r.T0 for r in ROOMS])
    n = np.array([10, 0, 5, 0, 0, 20])
    u = np.array([0.5, 0, 1.0, 0, 0.2, 0])
    out = step_temps(T, 30.0, n, u, NOMINAL)
    for i, r in enumerate(ROOMS):
        exp = T[i] + DT_H / r.C * ((30.0 - T[i]) / r.R + Q_OCC_KW * n[i] - 3.0 * r.P_ac * u[i])
        assert out[i] == pytest.approx(exp)


def test_07_energy_accounting_equation_3():
    cfg = PlannerConfig()
    res = run_episode(make_scenario("heat", 1), make_policy("threshold", cfg), cfg)
    E = 0.0
    for s in res.steps:
        p = power_kw(np.array(s["u"]), np.array(s["open"], dtype=float), NOMINAL)
        E += DT_H * p.sum()
    assert res.metrics["kwh"] == pytest.approx(E, abs=1e-3)


def test_08_occupied_deviation_uses_interval_midpoint():
    assert deviation(23.0) == pytest.approx(1.0)
    assert deviation(28.5) == pytest.approx(1.5)
    assert deviation(25.0) == 0.0
    log = {"E": 0.0, "V": 2 * deviation(0.5 * (27.0 + 28.0)) * DT_H, "occ_h": 2 * DT_H, "served": 2, "demand": 2,
           "unserved": 0, "M": 0, "K": 0, "LS": 0, "T_end": [28.0], "counts": [2]}
    m = episode_metrics([log])
    assert m["dev_c"] == pytest.approx(0.5)     # midpoint 27.5 → 0.5 °C, not 1.0 from the end temperature


def test_09_score_is_lexicographic_unserved_first():
    cfg = PlannerConfig()
    a = RolloutResult("a", "mask", U=10.0, E=1.0)
    b = RolloutResult("b", "mask", U=0.0, E=100.0, V=5.0)
    assert score_of(b, cfg).key() < score_of(a, cfg).key()
    assert Score(0.0, 1.0).key() < Score(0.0, 2.0).key()


def test_10_stale_snapshot_rejected():
    cfg = PlannerConfig()
    env = CampusEnv(make_scenario("steady", 0), cfg)
    s = env.get_state()
    a = mask_action(frozenset({"A101", "A102"}), s.health, cfg)
    r = env.apply_plan(a, s.version - 1, "stale")
    assert not r.executed and r.rejected[0]["code"] == "stale_snapshot"
    assert env.apply_plan(a, s.version, "fresh").executed


def test_11_idempotency_key_executes_once():
    cfg = PlannerConfig()
    env = CampusEnv(make_scenario("steady", 0), cfg)
    s = env.get_state()
    a = mask_action(frozenset({"A101", "A102", "A103"}), s.health, cfg)
    assert env.apply_plan(a, s.version, "k").executed
    before = env.ledger.counts().copy()
    r2 = env.apply_plan(mask_action(frozenset({"B201"}), s.health, cfg), s.version, "k")
    assert r2.duplicate and not r2.executed
    assert np.array_equal(env.ledger.counts(), before)


class _NoAckLayer(DirectDeviceLayer):
    def __init__(self, drop_room: str):
        super().__init__()
        self.drop_room = drop_room

    def command(self, cmds, now_min):
        out = super().command(cmds, now_min)
        out[self.drop_room] = None
        return out


def test_12_ledger_changes_only_after_ack():
    cfg = PlannerConfig()
    env = CampusEnv(make_scenario("steady", 0), cfg, _NoAckLayer("A101"))
    s = env.get_state()
    r = env.apply_plan(mask_action(frozenset({"A101", "B201"}), s.health, cfg), s.version, "k")
    assert r.outcomes["A101"] == "no_ack"
    c = env.ledger.counts()
    assert c[ROOM_IDS.index("A101")] == 0 and c[ROOM_IDS.index("B201")] > 0
    assert not env.get_state().open[ROOM_IDS.index("A101")]


@pytest.mark.parametrize("policy", ["fixed", "threshold", "threshold_eff", "planner", "arm_B"])
def test_13_episode_invariants_and_determinism(policy):
    cfg = PlannerConfig()
    sc = make_scenario("fault", 0, mismatch=True)
    r1 = run_episode(sc, make_policy(policy, cfg), cfg)
    r2 = run_episode(make_scenario("fault", 0, mismatch=True), make_policy(policy, cfg), cfg)
    m = r1.metrics
    assert m["hard_violations"] == 0
    assert 0 <= m["service_pct"] <= 100
    assert r1.metrics == r2.metrics
    for s in r1.steps:
        assert sum(s["counts"]) <= s["demand"]
        assert all(n <= c for n, c in zip(s["counts"], SEATS))
