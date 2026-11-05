"""The budgeted ``call`` path every role uses (no SDK, no network): a reply slower than the
step's remaining budget is abandoned at the deadline but its tokens are still counted when it
completes (they are billed), and any exception from a client is a failed call, not a crash."""
from __future__ import annotations

import time

from campusflex.llm.client import ABANDONED, AbandonedUsage, CallBudget, LLMResponse, call


class _Slow:
    """Answers correctly, but only after ``delay`` seconds (like a live reply over budget)."""
    model = "slow"

    def __init__(self, delay: float):
        self.delay = delay

    def complete_json(self, *, role, system, user, schema, effort=None):
        time.sleep(self.delay)
        return LLMResponse(role, {"ok": True}, model=self.model, latency_s=self.delay,
                           input_tokens=120, output_tokens=8)


def test_slow_reply_hits_the_wall_clock_deadline_and_its_tokens_are_kept():
    before = ABANDONED.snapshot()
    r = call(_Slow(0.3), CallBudget(3, 0.1), role="L3", system="s", user="u", schema={"type": "object"})
    assert r.error == "timeout" and not r.ok
    time.sleep(0.5)
    d = AbandonedUsage.delta(ABANDONED.snapshot(), before)
    assert d["calls"] == 1 and d["completed_late"] == 1 and d["input_tokens"] == 120


def test_a_reply_within_the_budget_is_returned():
    r = call(_Slow(0.01), CallBudget(3, 20.0), role="L3", system="s", user="u", schema={"type": "object"})
    assert r.ok and r.data == {"ok": True}


def test_a_client_exception_is_a_failed_call_not_a_crash():
    class Broken:
        model = "broken"

        def complete_json(self, **_):
            raise KeyError("boom")

    for budget in (None, CallBudget(3, 20.0), CallBudget(10, float("inf"))):     # E4 uses an unbounded budget
        r = call(Broken(), budget, role="L2", system="s", user="u", schema={"type": "object"})
        assert not r.ok and r.error.startswith("exception: KeyError")
