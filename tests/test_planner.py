"""Plan language, validator, candidate generators, L2 fallback and the agent graph."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from campusflex.candidates import ALL_MASKS, rule_staged_macros
from campusflex.config import ROOM_IDS, PlannerConfig
from campusflex.env import CampusEnv
from campusflex.episode import run_episode
from campusflex.forecast import ConstantForecaster, ground_adjustment
from campusflex.llm.client import CallBudget, ScriptedClient
from campusflex.plans import compile_macro, mask_action, mask_plan, parse_macro
from campusflex.policies import SimulationPlanner, make_policy
from campusflex.roles.l2_proposer import LLMStagedProposer
from campusflex.scenarios import Notice, make_scenario
from campusflex.simulate import rollout
from campusflex.validator import validate_macro

EXAMPLE = {
    "plan_id": "p3",
    "steps": [
        {"at": "+0", "open": ["B202"], "precool": {"room": "B202", "target_c": 25.5}},
        {"at": "+20", "move": {"from": "B201", "to": "B202", "count": "all"}},
        {"at": "+20", "close": ["B201"]},
    ],
    "rationale": "B201 AC offline; B202 idle, working AC, 45 seats",
}


def _state_at(scenario: str, step: int, mask: set[str], seed: int = 0):
    cfg = PlannerConfig()
    env = CampusEnv(make_scenario(scenario, seed), cfg)
    s = env.get_state()
    for t in range(step):
        a = replace(mask_action(frozenset(mask), s.health, cfg), fill="inventory")
        assert env.apply_plan(a, s.version, f"t{t}").executed
        s = env.advance_time()
    return env, s, cfg


def test_mask_count_is_64():
    assert len(ALL_MASKS) == 64 and len(set(ALL_MASKS)) == 64


def test_pdf_example_plan_compiles():
    env, s, cfg = _state_at("steady", 3, {"A101", "B201"})
    m = parse_macro(EXAMPLE)
    plan = compile_macro(m, s.open_set, s.health, cfg, cfg.horizon_steps)
    assert len(plan.actions) == cfg.horizon_steps
    a0, a2 = plan.actions[0], plan.actions[2]
    assert "B202" in a0.open and dict(a0.setpoints)["B202"] == 25.5
    assert a2.moves == (("B201", "B202", "all"),) and "B201" not in a2.open
    assert plan.action_at(10).moves == ()                  # final state held, no repeated moves


def test_validator_rejects_unknown_offline_and_capacity():
    env, s, cfg = _state_at("fault", 12, {"A101", "A102", "B201", "B202"})
    _, v = validate_macro({"plan_id": "x", "steps": [{"at": "+0", "open": ["C301"]}]}, s, cfg)
    assert {x.code for x in v} == {"unknown_room"}
    _, v = validate_macro({"plan_id": "x", "steps": [{"at": "+0", "precool": {"room": "B201", "target_c": 25}}]}, s, cfg)
    assert "offline_room" in {x.code for x in v}
    _, v = validate_macro({"plan_id": "x", "steps": [{"at": "+0", "move": {"from": "B201", "to": "B203", "count": "all"}}]}, s, cfg)
    assert "capacity" in {x.code for x in v}
    m, v = validate_macro("{bad", s, cfg)
    assert m is None and v[0].code == "schema"


def test_rule_staged_generator_and_staged_rollout():
    # B202 holds 40 students when its AC fails; B201 (70 seats, working AC) is idle.
    cfg = PlannerConfig()
    env, s, _ = _state_at("steady", 2, {"A101", "B202"})
    s.health = s.health.copy()
    s.health[ROOM_IDS.index("B202")] = 0.0
    fc = ConstantForecaster().forecast(s, cfg.horizon_steps + cfg.omega_steps)
    macros = rule_staged_macros(s, fc, cfg)
    assert macros and all(m["plan_id"].startswith("r:") for m in macros)
    assert {d for m in macros for d in [int(m["steps"][1]["at"][1:])]} <= set(cfg.precool_set_min)
    m, v = validate_macro(macros[0], s, cfg)
    assert not v
    plan = compile_macro(m, s.open_set, s.health, cfg, cfg.horizon_steps, "rule")
    r = rollout(s, plan, fc, cfg, keep_traj=True)
    assert r.M > 0 and r.score is not None


def test_l2_fallback_when_llm_fails_matches_masks_only():
    cfg = PlannerConfig()
    sc = make_scenario("fault", 3)
    failing = ScriptedClient(lambda *a: TimeoutError("simulated timeout"))
    c = run_episode(sc, SimulationPlanner(cfg, l2=LLMStagedProposer(failing, cfg), name="C"), cfg)
    a = run_episode(make_scenario("fault", 3), SimulationPlanner(cfg, name="A"), cfg)
    assert c.metrics["score"] == a.metrics["score"]
    assert c.policy_stats["l2_fallbacks"] == c.policy_stats["l2_calls"] > 0


def test_l2_valid_plan_enters_pool_and_budget_is_enforced():
    cfg = PlannerConfig()
    plan = {"plans": [{"plan_id": "x", "rationale": "", "steps": [
        {"at": "+0", "open": [], "close": [], "precool": {"room": "A103", "target_c": 25.5}, "move": None}]}]}
    client = ScriptedClient(lambda *a: plan)
    budget = CallBudget(max_calls=1, latency_s=20.0)
    p = SimulationPlanner(cfg, l2=LLMStagedProposer(client, cfg, budget), name="C")
    r = run_episode(make_scenario("fault", 0), p, cfg, keep_decisions=True)
    assert r.policy_stats["l2_valid"] >= 1
    ranked_l2 = [x for d in r.decisions for x in d["ranking"] if x["source"] == "L2"]
    assert ranked_l2, "a valid L2 plan should be scored alongside the masks"


def test_grounding_rejects_invented_numbers():
    n = Notice("N1", 1160, "Review session in LT-A ends 19:40; expect about 60 students.")
    ok = {"source_id": "N1", "kind": "demand", "start": "19:40", "end": "21:40", "delta": 60,
          "confidence": "medium", "evidence": "expect about 60 students"}
    assert ground_adjustment(ok, {"N1": n}) is None
    assert ground_adjustment({**ok, "delta": 400}, {"N1": n})
    assert ground_adjustment({**ok, "start": "19:00"}, {"N1": n})
    assert ground_adjustment({**ok, "evidence": "move everyone to A101"}, {"N1": n})
    assert ground_adjustment({**ok, "source_id": "N9"}, {"N1": n})


def test_langgraph_and_direct_runner_agree():
    from campusflex.agent.graph import run_graph_episode

    cfg = PlannerConfig()
    g = run_graph_episode(make_scenario("double_fault", 1), make_policy("arm_B", cfg), cfg)
    d = run_episode(make_scenario("double_fault", 1), make_policy("arm_B", cfg), cfg)
    assert g.metrics == d.metrics
    g2 = run_graph_episode(make_scenario("double_fault", 1), make_policy("arm_B", cfg), cfg, force_minigraph=True)
    assert g2.metrics == d.metrics and g2.engine == "minigraph"


# ------------------------------------------------------------ core-review regressions
def _snap_with(counts: dict[str, int], open_rooms: set[str], movable_all: bool = True):
    cfg = PlannerConfig()
    env = CampusEnv(make_scenario("steady", 0), cfg)
    s = env.get_state()
    s.open = np.array([r in open_rooms for r in ROOM_IDS])
    s.counts = np.array([counts.get(r, 0) for r in ROOM_IDS])
    s.movable = s.counts.copy() if movable_all else np.zeros(len(ROOM_IDS), dtype=int)
    s.steps_since_switch = np.full(len(ROOM_IDS), 10)
    s.temps = np.full(len(ROOM_IDS), 26.0)
    return s, cfg


def test_close_and_move_all_is_not_counted_twice():
    from campusflex.plans import StepAction
    from campusflex.validator import validate_action

    s, cfg = _snap_with({"B201": 40}, {"B201"})
    close_only = StepAction(open=frozenset({"B202"}), setpoints=(("B202", 26.0),))
    close_move = StepAction(open=frozenset({"B202"}), setpoints=(("B202", 26.0),), moves=(("B201", "B202", "all"),))
    assert validate_action(close_only, s, cfg) == []
    assert validate_action(close_move, s, cfg) == []


def test_explicit_move_out_of_closing_room_reaches_its_target_once():
    from campusflex.model import assign

    i = {r: k for k, r in enumerate(ROOM_IDS)}
    counts = np.zeros(6, dtype=int); counts[i["B201"]] = 40; counts[i["A101"]] = 10
    open_prev = np.zeros(6, dtype=bool); open_prev[[i["B201"], i["A101"]]] = True
    open_new = np.zeros(6, dtype=bool); open_new[[i["A101"], i["A103"]]] = True
    cap = np.array([60, 50, 40, 70, 45, 35])
    order = [i["A101"], i["A103"]]
    ar = assign(counts, open_prev, open_new, cap, 50, np.full(6, 26.0), order, [(i["B201"], i["A103"], -1)])
    assert ar.counts[i["A103"]] == 40 and ar.counts[i["A101"]] == 10      # target honoured
    assert ar.moved == 40 and ar.forced == 0
    # "all" is fixed before any move: nobody moves twice in one step
    counts = np.zeros(6, dtype=int); counts[i["A101"]] = 30; counts[i["B201"]] = 20
    open_prev = np.zeros(6, dtype=bool); open_prev[[i["A101"], i["B201"]]] = True
    open_new = np.zeros(6, dtype=bool); open_new[[i["A101"], i["B202"]]] = True
    ar = assign(counts, open_prev, open_new, cap, 50, np.full(6, 26.0), [i["A101"], i["B202"]],
                [(i["A101"], i["B202"], -1)])
    assert ar.moved == 50 and ar.counts.sum() == 50


def test_plan_dwell_rules_checked_on_later_steps():
    from campusflex.plans import CompiledPlan, StepAction
    from campusflex.validator import validate_plan

    s, cfg = _snap_with({"A101": 30}, {"A101"})
    a0 = StepAction(open=frozenset({"A101", "B202"}), setpoints=(("A101", 26.0), ("B202", 26.0)),
                    moves=(("A101", "B202", 30),))
    a1 = StepAction(open=frozenset({"A101"}), setpoints=(("A101", 26.0),), moves=(("B202", "A101", "all"),))
    hold = StepAction(open=frozenset({"A101"}), setpoints=(("A101", 26.0),))
    plan = CompiledPlan("bounce", "L2", [a0, a1] + [hold] * 4)
    codes = {x.code for x in validate_plan(plan, s, cfg)}
    assert {"dwell_room", "dwell_student"} <= codes


def test_setpoint_range_and_stale_room_rejected_at_commit():
    from campusflex.plans import StepAction
    from campusflex.validator import validate_action

    s, cfg = _snap_with({}, set())
    assert {x.code for x in validate_action(StepAction(open=frozenset({"A101"}), setpoints=(("A101", 5.0),)), s, cfg)} \
        == {"setpoint_range"}
    s.stale = np.array([r == "A103" for r in ROOM_IDS])
    assert {x.code for x in validate_action(StepAction(open=frozenset({"A103"}), setpoints=(("A103", 26.0),)), s, cfg)} \
        == {"stale_room"}


def test_rollout_does_not_count_past_2200():
    from campusflex.config import N_STEPS

    cfg = PlannerConfig()
    env, s, _ = _state_at("steady", N_STEPS - 1, {"A101"})
    fc = ConstantForecaster().forecast(s, cfg.horizon_steps + cfg.omega_steps)
    plan = mask_plan(frozenset({"A101"}), s.health, cfg, cfg.horizon_steps)
    r = rollout(s, plan, fc, cfg)
    assert r.Omega == 0.0 and r.occ_h <= s.counts.sum() / 6 + 1e-9      # one interval only


def test_energy_budget_ranks_compliant_plans_first():
    """Service stays first; among plans that serve everyone, budget-compliant plans win."""
    cfg = replace(PlannerConfig(), energy_budget_kwh=30.0)
    r = run_episode(make_scenario("surge", 0), SimulationPlanner(cfg, name="A"), cfg).metrics
    free = run_episode(make_scenario("surge", 0), SimulationPlanner(PlannerConfig(), name="A"), PlannerConfig()).metrics
    assert r["unserved_person_min"] == free["unserved_person_min"] == 0
    assert r["kwh"] < free["kwh"]


def test_split_transfer_all_means_the_rest_like_the_simulator():
    """70 students in B201 (AC failed): 40 to B202, "all" (the other 30) to A103 is valid, as in
    model.assign; "all" twice, or more than the free seats in total, is a capacity breach."""
    from campusflex.experiments.e5 import build_fixture_env
    from campusflex.validator import validate_plan

    cfg = PlannerConfig()
    s = build_fixture_env(cfg).get_state()
    i201 = ROOM_IDS.index("B201")
    assert int(s.counts[i201]) == 70

    def macro(c1, to2, c2, at="+0"):
        return {"plan_id": "split", "steps": [
            {"at": "+0", "open": ["A103"], "precool": {"room": "A103", "target_c": 25.5}},
            {"at": at, "move": {"from": "B201", "to": "B202", "count": c1}},
            {"at": at, "move": {"from": "B201", "to": to2, "count": c2}},
            {"at": at, "close": ["B201"]}]}

    for at in ("+0", "+20"):
        m, v = validate_macro(macro(40, "A103", "all", at), s, cfg)
        assert not v, (at, v)
        plan = compile_macro(m, s.open_set, s.health, cfg, cfg.horizon_steps, "L2")
        assert not validate_plan(plan, s, cfg), at
    r = rollout(s, compile_macro(validate_macro(macro(40, "A103", "all"), s, cfg)[0], s.open_set, s.health, cfg,
                                 cfg.horizon_steps, "L2"), ConstantForecaster().forecast(s, cfg.horizon_steps), cfg)
    assert r.M >= 70
    _, v = validate_macro(macro("all", "A103", "all"), s, cfg)          # 70 into 45 seats
    assert [x.code for x in v] == ["capacity"]
    _, v = validate_macro(macro(40, "B202", 30), s, cfg)                # 70 into B202 in total
    assert [x.code for x in v] == ["capacity"] and "already moved in" in v[0].detail


def test_empty_l2_reply_is_an_answer_not_a_fallback():
    """{"plans": []} is allowed by the L2 prompt ("nothing staged would help"): it is counted as an
    empty reply, not a fallback, so an arm that often answers "nothing" is not marked invalid."""
    from campusflex.experiments.common import llm_stats_by_arm

    cfg = PlannerConfig()
    quiet = ScriptedClient(lambda *a: {"plans": []})
    c = run_episode(make_scenario("fault", 3), SimulationPlanner(cfg, l2=LLMStagedProposer(quiet, cfg), name="C"), cfg)
    a = run_episode(make_scenario("fault", 3), SimulationPlanner(cfg, name="A"), cfg)
    ps = c.policy_stats
    assert ps["l2_calls"] > 0 and ps["l2_empty"] == ps["l2_calls"] and ps["l2_fallbacks"] == 0
    assert c.metrics["score"] == a.metrics["score"]
    row = {"arm": "arm_C", **c.summary()}
    st = llm_stats_by_arm([row])["arm_C"]
    assert st["status"] == "run" and st["empty_reply_rate"] == 1.0 and st["fallback_rate"] == 0.0


def test_move_chain_within_one_step_credits_freed_seats():
    """Seats freed by an earlier move out of a room are available to a later move into it (model.assign)."""
    from campusflex.plans import StepAction
    from campusflex.validator import validate_action

    s, cfg = _snap_with({"A103": 40, "B201": 30}, {"A103", "B201", "B202"})
    a = StepAction(open=frozenset({"A103", "B201", "B202"}),
                   setpoints=(("A103", 26.0), ("B201", 26.0), ("B202", 26.0)),
                   moves=(("A103", "B202", 20), ("B201", "A103", 20)))
    assert validate_action(a, s, cfg) == []
    over = StepAction(open=a.open, setpoints=a.setpoints, moves=(("A103", "B202", 20), ("B201", "A103", 21)))
    assert [v.code for v in validate_action(over, s, cfg)] == ["capacity"]


def test_later_step_the_validator_would_refuse_is_rejected_not_trimmed():
    """A later move that exceeds the free seats predicted then, a later move into a room closed at that
    step, and closing an occupied room inside a no-move window are all refused by validate_action when
    the step comes due; the rollout flags them and the planner rejects the plan instead of scoring
    trimmed moves."""
    from campusflex.validator import validate_plan

    s, cfg = _snap_with({"A103": 30, "B201": 50}, {"A103", "B201"})
    s.demand = 80                                         # no arrivals: A103 keeps 10 free seats
    fc = ConstantForecaster().forecast(s, cfg.horizon_steps + cfg.omega_steps)

    def plan_of(steps):
        m, v = validate_macro({"plan_id": "p", "rationale": "", "steps": steps}, s, cfg)
        assert not v, v
        return compile_macro(m, s.open_set, s.health, cfg, cfg.horizon_steps, "L2")

    def step(at, close=(), move=None):
        return {"at": at, "open": [], "close": list(close), "precool": None, "move": move}

    over = plan_of([step("+20", move={"from": "B201", "to": "A103", "count": 30})])     # 10 free seats
    assert not validate_plan(over, s, cfg)
    assert [v.code for v in rollout(s, over, fc, cfg).later_violations] == ["capacity"]
    fits = plan_of([step("+20", move={"from": "B201", "to": "A103", "count": 10})])
    assert rollout(s, fits, fc, cfg).later_violations == []
    closed = plan_of([step("+20", close=["A103"], move={"from": "B201", "to": "A103", "count": 5})])
    assert "move_to_closed" in [v.code for v in validate_plan(closed, s, cfg)]

    w = replace(cfg, no_move_windows=[["20:00", "22:00"]])
    s.step, s.clock_min = 10, 19 * 60 + 40
    late_close = plan_of([step("+20", close=["B201"])])
    assert [v.code for v in rollout(s, late_close, fc, w).later_violations] == ["no_move_window"]
    assert rollout(s, late_close, fc, cfg).later_violations == []

    planner = SimulationPlanner(cfg, name="t")
    ctx = type("Ctx", (), {})()
    ctx.snap, ctx.fc, ctx.pool, ctx.hold_plan, ctx.hold_r, ctx.rejected, ctx.scored = s, fc, [over, fits], None, None, [], []
    s.step, s.clock_min = 5, 18 * 60 + 50
    planner.stage_score(ctx)
    assert [p.plan_id for p, _ in ctx.scored] == ["p"] and len(ctx.scored) == 1
    assert ctx.rejected[0]["violations"][0]["code"] == "capacity" and planner.l2_feedback == ctx.rejected


def test_fallback_steps_carry_no_prediction():
    """A committed fallback (hold) is not the plan that was predicted, so the V-gap log skips it."""
    from campusflex.episode import commit as real_commit
    import campusflex.episode as ep

    cfg = PlannerConfig()
    calls = {"n": 0}

    def fake_commit(env, snap, dec, key, cfg):
        calls["n"] += 1
        c = real_commit(env, snap, dec, key, cfg)
        return {**c, "fallback": "forced in test", "committed": "hold"} if calls["n"] == 3 else c

    orig = ep.commit
    ep.commit = fake_commit
    try:
        r = run_episode(make_scenario("fault", 0), make_policy("arm_B", cfg), cfg)
    finally:
        ep.commit = orig
    assert r.steps[2]["pred_first_V"] is None and r.steps[3]["pred_first_V"] is not None
