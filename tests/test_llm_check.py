"""``python -m campusflex llm-check``: per-role readiness report and E1 C/D projection (no API)."""
from __future__ import annotations

import re

import pytest

from campusflex import __main__ as cli
from campusflex.config import PlannerConfig
from campusflex.experiments import common, llm_check
from campusflex.llm.client import LLMResponse, ScriptedClient

CFG = PlannerConfig()


def good_reply(role: str, system: str, user: str):
    if role == "L1":
        return {"interval": {"start": "18:00", "end": "22:00"}, "demand_estimate": 120,
                "band_low_c": 24.0, "band_high_c": 27.0, "target_c": 26.0, "energy_budget_kwh": None,
                "power_cap_kw": None, "relocation": {"max_moves_per_student": None, "no_move_windows": []},
                "weight_levels": {"lambda_M": "high", "lambda_T": "default", "lambda_S": "default"},
                "defaults_filled": ["interval", "band", "target_c"], "conflicts": [], "clarifying_questions": []}
    if role == "L2":       # A103 is closed with a working AC at the fixture state
        return {"plans": [{"plan_id": "p1", "steps": [{"at": "+0", "open": ["A103"], "close": [],
                                                       "precool": {"room": "A103", "target_c": 25.5},
                                                       "move": None}]}]}
    if role == "L3":
        return {"adjustments": [{"source_id": "N1", "kind": "demand", "start": "20:30", "end": "22:00",
                                 "delta": 60, "confidence": "medium", "evidence": "expect about 60 students",
                                 "recheck_room": None}]}
    if role == "L4":
        return {"explanation": "Option 1 keeps every student seated; the alternatives use more energy."}
    if role == "L5":
        interval = re.search(r'"(fault/s0/[^"]+)"', user).group(1)
        return {"review_id": "R1",
                "evidence": [{"interval": interval, "observation": "occupants left in a warming room"}],
                "diagnosis": "terminal term too weak",
                "diff": {"param": "lambda_Omega", "from": CFG.lambda_Omega, "to": 8.0},
                "expected_effect": {"metric": "V", "direction": "decrease", "scenarios": ["fault"]},
                "risk": "more pre-cooling energy"}
    raise AssertionError(role)


@pytest.fixture
def frozen_here(tmp_path, monkeypatch):
    from campusflex.experiments import freeze

    monkeypatch.setattr(common, "FROZEN", tmp_path / "frozen_config.json")
    freeze.freeze(tmp_path / "frozen_config.json")
    return tmp_path


def test_ready_when_every_role_answers_usably(frozen_here):
    res = llm_check.run(ScriptedClient(good_reply, model="deepseek-flash"), repeats=2,
                        prices={"in": 1.0, "out": 2.0}, out=None)
    assert res["status"] == "ready", res["warnings"]
    for role in llm_check.ROLES:
        s = res["roles"][role]
        assert s["ok_calls"] == s["usable_outputs"] == 2, (role, s)
    assert res["roles"]["L2"]["valid_plans_mean"] == 1.0
    p = res["projection_e1_cd"]
    assert p["episodes"] == 480 and p["l2_calls_est"] > 0 and "cost_usd_est" in p
    assert res["meta"]["prompts_frozen"] is True


def test_warns_when_l2_exceeds_step_budget(frozen_here, monkeypatch):
    monkeypatch.setattr(llm_check, "count_e1_cd_l2_calls",
                        lambda cfg: {"l2_calls_per_episode_mean": 10.0, "episodes": 480, "l2_calls_est": 4800,
                                     "measured_on": "stub"})

    def slow(role, system, user):
        if role == "L2":
            return LLMResponse("L2", good_reply(role, system, user), model="deepseek-flash", latency_s=25.0)
        return good_reply(role, system, user)

    res = llm_check.run(ScriptedClient(slow, model="deepseek-flash"), repeats=1, roles=("L2",), out=None)
    s = res["roles"]["L2"]
    assert s["errors"] == {"latency_exceeded": 1} and s["within_step_budget"] == 0
    assert res["status"] == "check_warnings"
    assert any("exceeds the 20 s step budget" in w for w in res["warnings"])
    assert res["projection_e1_cd"]["wall_hours_est"] > 0


def test_unusable_replies_and_unfrozen_prompts_are_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "FROZEN", tmp_path / "missing.json")
    res = llm_check.run(ScriptedClient(lambda *_: {}, model="deepseek-flash"), repeats=1, roles=("L2", "L3"),
                        projection=False, out=None)
    assert res["roles"]["L2"]["usable_outputs"] == 0 and "projection_e1_cd" not in res
    w = " | ".join(res["warnings"])
    assert "L2: 1/1 replies were not usable" in w and "freeze gate" in w


def test_cli_without_key_exits_nonzero(tmp_path, monkeypatch):
    assert cli.COMMANDS["llm-check"] == "campusflex.experiments.llm_check"
    monkeypatch.setattr(common, "llm_client", lambda **kw: (None, "RuntimeError: DEEPSEEK_API_KEY is not set"))
    with pytest.raises(SystemExit) as e:
        llm_check.main(["--out", str(tmp_path / "c.json")])
    assert "DEEPSEEK_API_KEY" in str(e.value)
    assert '"not_run"' in (tmp_path / "c.json").read_text()


def test_occasional_bad_reply_is_a_note_but_account_errors_and_majorities_warn(frozen_here):
    from campusflex.llm.client import LLMResponse

    roles = {"L1": {"n": 3, "ok_calls": 2, "usable_outputs": 2, "errors": {"invalid_json: x": 1}, "latency_s": {}},
             "L3": {"n": 3, "ok_calls": 1, "usable_outputs": 1, "errors": {"timeout": 2}, "latency_s": {}},
             "L5": {"n": 3, "ok_calls": 0, "usable_outputs": 0,
                    "errors": {"api_status_402: Insufficient Balance": 3}, "latency_s": {}}}
    warn, notes = llm_check.verdict(roles, 20.0, {"prompts_frozen": True})
    assert notes == ["L1: 1/3 calls failed ({'invalid_json: x': 1})"]
    assert len(warn) == 2 and warn[0].startswith("L3: 2/3") and "account error" in warn[1]
    flaky = iter([LLMResponse("L1", None, error="invalid_json: cut"), None, None])

    def reply(role, system, user):
        return next(flaky, None) or good_reply(role, system, user)

    res = llm_check.run(ScriptedClient(reply, model="deepseek-flash"), repeats=3, roles=("L1",), out=None,
                        projection=False)
    assert res["status"] == "ready" and res["notes"] and not res["warnings"]
