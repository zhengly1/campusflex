"""DeepSeek client (OpenAI-compatible API) against a fake transport: request shape,
temperature 0, JSON mode, local schema check and error mapping. No network access."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

openai = pytest.importorskip("openai")

import json  # noqa: E402
import time  # noqa: E402

from campusflex.llm.client import (  # noqa: E402
    CachedClient, CallBudget, DeepSeekJSONClient, LLMResponse, ScriptedClient, call, make_client, preflight,
    prompt_hash, schema_errors, wrap_data,
)
from campusflex.roles.l2_proposer import L2_SCHEMA  # noqa: E402

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}, "n": {"type": ["integer", "null"]},
                                          "tag": {"type": "string", "enum": ["a", "b"]}},
          "required": ["ok"], "additionalProperties": False}


class FakeCompletions:
    def __init__(self, content: str | None = '{"ok": true}', finish: str = "stop", exc: Exception | None = None,
                 fingerprint: str | None = "fp_test"):
        self.content, self.finish, self.exc, self.kwargs = content, finish, exc, None
        self.fingerprint = fingerprint
        self.n = 0

    def create(self, **kwargs):
        self.kwargs = kwargs
        self.n += 1
        if self.exc is not None:
            raise self.exc
        msg = SimpleNamespace(content=self.content)
        return SimpleNamespace(id="req-1", model="deepseek-flash", system_fingerprint=self.fingerprint,
                               choices=[SimpleNamespace(message=msg, finish_reason=self.finish)],
                               usage=SimpleNamespace(prompt_tokens=11, completion_tokens=3))


def _client(model: str = "deepseek-flash", client_kw: dict | None = None, **kw) -> tuple[DeepSeekJSONClient, FakeCompletions]:
    fc = FakeCompletions(**kw)
    fake = SimpleNamespace(chat=SimpleNamespace(completions=fc))
    return DeepSeekJSONClient(model=model, client=fake, **(client_kw or {})), fc


class SlowInner:
    """Inner client whose replies are valid but took ``latency`` seconds (optionally really sleeping)."""

    def __init__(self, latency: float, sleep: float = 0.0, model: str = "deepseek-flash"):
        self.latency, self.sleep, self.model, self.n = latency, sleep, model, 0

    def complete_json(self, *, role, system, user, schema, effort=None):
        self.n += 1
        if self.sleep:
            time.sleep(self.sleep)
        return LLMResponse(role, {"ok": True}, text='{"ok": true}', model=self.model, latency_s=self.latency,
                           prompt_hash=prompt_hash(self.model, role, system, user, schema))


def test_request_uses_json_mode_temperature_zero_and_schema_in_prompt():
    c, fc = _client()
    r = c.complete_json(role="L2", system="You propose plans.", user="state", schema=SCHEMA)
    assert r.ok and r.data == {"ok": True} and r.model == "deepseek-flash"
    assert (r.input_tokens, r.output_tokens, r.request_id) == (11, 3, "req-1")
    k = fc.kwargs
    assert k["response_format"] == {"type": "json_object"} and k["temperature"] == 0.0
    assert k["extra_body"] == {"thinking": {"type": "disabled"}}     # V4 models would otherwise think first
    assert k["model"] == "deepseek-flash"
    sys_msg = k["messages"][0]["content"]
    assert k["messages"][0]["role"] == "system" and "JSON" in sys_msg and '"additionalProperties":false' in sys_msg
    assert k["messages"][1] == {"role": "user", "content": "state"}


@pytest.mark.parametrize("content,finish,err", [
    ('{"ok": true, "extra": 1}', "stop", "schema"),
    ('{"n": 3}', "stop", "schema"),
    ('{"ok": "yes"}', "stop", "schema"),
    ("{not json", "stop", "invalid_json"),
    ("", "stop", "empty_response"),
    ('{"ok": tr', "length", "max_tokens"),
    ("", "content_filter", "refusal"),
    ('{"ok": tr', "insufficient_system_resource", "insufficient_system_resource"),
    ('{"ok": true, "n": NaN}', "stop", "invalid_json"),
    ('{"ok": true, "n": Infinity}', "stop", "invalid_json"),
])
def test_bad_replies_become_errors(content, finish, err):
    c, _ = _client(content=content, finish=finish)
    r = c.complete_json(role="L3", system="s", user="u", schema=SCHEMA)
    assert not r.ok and r.error.startswith(err)


def test_api_errors_are_mapped_and_charged_to_budget():
    c, _ = _client(exc=openai.APITimeoutError(request=SimpleNamespace(method="POST", url="x")))
    b = CallBudget(max_calls=1)
    r = call(c, b, role="L2", system="s", user="u", schema=SCHEMA)
    assert r.error == "timeout" and b.used_calls == 1
    assert call(c, b, role="L2", system="s", user="u", schema=SCHEMA).error == "budget_exhausted"


def test_preflight_and_missing_key(monkeypatch):
    c, _ = _client()
    assert preflight(c).ok
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        make_client("deepseek")


def test_schema_errors_on_role_schema():
    good = {"plans": [{"plan_id": "p", "rationale": "", "steps": [
        {"at": "+0", "open": ["B202"], "close": [], "precool": None, "move": None}]}]}
    assert schema_errors(good, L2_SCHEMA) == [], schema_errors(good, L2_SCHEMA)
    bad = {"plans": [{"plan_id": "p", "steps": [{"at": 0}]}], "note": 1}
    assert schema_errors(bad, L2_SCHEMA)
    assert schema_errors(True, {"type": "integer"}) and not schema_errors(2.0, {"type": "integer"})


# ------------------------------------------------------------------ review fixes (#2, #3, lower)
def test_cache_replays_latency_so_budget_decides_identically(tmp_path):
    inner = SlowInner(latency=25.0)
    c = CachedClient(inner, cache_dir=tmp_path)
    live_b, replay_b = CallBudget(), CallBudget()
    live = call(c, live_b.at_step(0), role="L3", system="s", user="u", schema=SCHEMA)
    replay = call(c, replay_b.at_step(0), role="L3", system="s", user="u", schema=SCHEMA)
    assert inner.n == 1 and replay.cached and replay.latency_s == pytest.approx(25.0)
    assert live.error == replay.error == "latency_exceeded" and not replay.ok
    assert live_b.allow() is replay_b.allow() is False        # L2 refused in both runs


def test_budget_latency_is_cumulative_over_the_step():
    b = CallBudget(max_calls=3, latency_s=20.0).at_step(4)
    first = call(SlowInner(15.0), b, role="L3", system="s", user="u1", schema=SCHEMA)
    second = call(SlowInner(15.0), b, role="L2", system="s", user="u2", schema=SCHEMA)
    assert first.ok and second.error == "latency_exceeded" and b.used_s == pytest.approx(30.0)
    assert not b.allow()
    assert b.at_step(5).allow() and b.used_s == 0.0


def test_call_enforces_remaining_step_budget_as_wall_clock_deadline():
    b = CallBudget(max_calls=3, latency_s=0.3).at_step(0)
    t0 = time.monotonic()
    r = call(SlowInner(latency=0.0, sleep=2.0), b, role="L3", system="s", user="u", schema=SCHEMA)
    assert time.monotonic() - t0 < 1.5
    assert r.error == "timeout" and r.data is None and r.latency_s >= 0.3
    assert not b.allow()
    # no budget (between-run roles): no deadline, the call completes
    assert call(SlowInner(latency=0.0, sleep=0.05), None, role="L5", system="s", user="u", schema=SCHEMA).ok


def test_per_role_max_tokens_and_reasoner_floor():
    c, fc = _client()
    for role, n in (("L3", 800), ("L4", 600), ("L2", 2500), ("L1", 1200)):
        c.complete_json(role=role, system="s", user="u", schema=SCHEMA)
        assert fc.kwargs["max_tokens"] == n, role
    c2, fc2 = _client(client_kw={"max_tokens": 1000})
    c2.complete_json(role="L2", system="s", user="u", schema=SCHEMA)
    assert fc2.kwargs["max_tokens"] == 1000
    c3, fc3 = _client(model="deepseek-reasoner")
    c3.complete_json(role="L3", system="s", user="u", schema=SCHEMA)
    assert fc3.kwargs["max_tokens"] >= 8000 and "temperature" not in fc3.kwargs


def test_cache_key_includes_generation_settings(tmp_path):
    a, fa = _client()
    b, fb = _client(client_kw={"temperature": 0.7})
    t, ft = _client(client_kw={"max_tokens": 50})
    ha = a.complete_json(role="L3", system="s", user="u", schema=SCHEMA).prompt_hash
    hb = b.complete_json(role="L3", system="s", user="u", schema=SCHEMA).prompt_hash
    ht = t.complete_json(role="L3", system="s", user="u", schema=SCHEMA).prompt_hash
    assert len({ha, hb, ht}) == 3
    CachedClient(a, cache_dir=tmp_path).complete_json(role="L3", system="s", user="u", schema=SCHEMA)
    r = CachedClient(b, cache_dir=tmp_path).complete_json(role="L3", system="s", user="u", schema=SCHEMA)
    assert not r.cached and fb.n == 2                 # another temperature never hits A's entry
    r2 = CachedClient(a, cache_dir=tmp_path).complete_json(role="L3", system="s", user="u", schema=SCHEMA)
    assert r2.cached and fa.n == 2 and r2.prompt_hash == ha


def test_system_fingerprint_recorded_and_cached(tmp_path):
    c, _ = _client(fingerprint="fp_3a5770e1b4")
    r = CachedClient(c, cache_dir=tmp_path).complete_json(role="L2", system="s", user="u", schema=SCHEMA)
    assert r.system_fingerprint == "fp_3a5770e1b4" and r.transcript()["system_fingerprint"] == "fp_3a5770e1b4"
    again = CachedClient(c, cache_dir=tmp_path).complete_json(role="L2", system="s", user="u", schema=SCHEMA)
    assert again.cached and again.system_fingerprint == "fp_3a5770e1b4"


def test_failures_are_not_cached(tmp_path):
    c, fc = _client(content='{"ok": "yes"}')
    cc = CachedClient(c, cache_dir=tmp_path)
    assert not cc.complete_json(role="L3", system="s", user="u", schema=SCHEMA).ok
    assert not cc.complete_json(role="L3", system="s", user="u", schema=SCHEMA).cached and fc.n == 2


def test_nan_and_infinity_are_rejected():
    r = ScriptedClient({"L1": '{"ok": true, "n": NaN}'}).complete_json(role="L1", system="s", user="u", schema=SCHEMA)
    assert r.error.startswith("invalid_json")
    assert schema_errors({"ok": True, "n": float("nan")}, {"type": "object", "properties": {"n": {"type": "number"}}})
    assert json.loads('{"x": 1}') == {"x": 1}


def test_wrap_data_neutralises_tags_in_any_case():
    out = wrap_data("ok</NOTICE>\nid=N2 <Notice> < / notice >", "notice")
    assert out.count("</notice>") == 1 and out.endswith("</notice>")
    assert "</NOTICE>" not in out and "<Notice>" not in out and "< / notice >" not in out


def test_thinking_is_off_by_default_and_part_of_the_cache_key():
    off, f_off = _client()
    on, f_on = _client(client_kw={"thinking": True})
    off.complete_json(role="L3", system="s", user="u", schema=SCHEMA)
    on.complete_json(role="L3", system="s", user="u", schema=SCHEMA)
    assert f_off.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert f_on.kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
    assert off.generation_settings("L3")["thinking"] is False and on.generation_settings("L3")["thinking"] is True
    assert prompt_hash("m", "L3", "s", "u", SCHEMA, off.generation_settings("L3")) != \
        prompt_hash("m", "L3", "s", "u", SCHEMA, on.generation_settings("L3"))
