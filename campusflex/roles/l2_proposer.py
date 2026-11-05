"""L2 staged-plan proposer (main contribution, Section 3.4).

On a trigger the LLM receives a compact state summary and proposes up to five macro-plans
in the plan language. It never sends device commands and never supplies a number that a
tool did not return: its plans are compiled, validated, simulated and scored with Eq. 4
exactly like the 64 masks. The rationale is logged, never scored. Rejection reasons from
the validator are returned to the LLM on its next call.
"""
from __future__ import annotations

import json
from typing import Any

from ..config import ROOM_IDS, STEP_MIN, PlannerConfig
from ..forecast import Forecast
from ..llm.client import CallBudget, LLMClient, call
from ..plans import MACRO_PLAN_JSON_SCHEMA
from ..policies import ProposalBatch
from ..state import Snapshot
from ..validator import PRECOOL_MIN_C

L2_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"plans": {"type": "array", "items": MACRO_PLAN_JSON_SCHEMA}},
    "required": ["plans"],
    "additionalProperties": False,
}

L2_SYSTEM = """You are the staged-plan proposer inside CampusFlex, a controller for six study rooms during an \
examination evening. A deterministic planner already evaluates every single room mask held for the whole \
horizon. Your job is to propose up to five *staged* macro-plans that a single held mask cannot express, \
for example: pre-cool an idle room, then transfer the occupants of a warming or faulty room into it.

How your output is used:
- Each plan is compiled, checked by a validator, simulated with a thermal model and scored. You do not \
  estimate energy, temperature or comfort; the simulator does. Your rationale is logged but never scored.
- Plans that name unknown rooms, rooms whose air conditioner is offline (as a destination), unavailable \
  rooms, capacity breaches, or offsets outside the horizon are rejected. Rejections from your previous call \
  are listed under "previous_rejections"; do not repeat them.
- Only the first step of the winning plan is committed; the rest is re-scored at the next step.

Plan language (JSON):
- plan_id: short id. steps: list of {"at": "+<minutes>", "open": [...], "close": [...], \
  "precool": {"room", "target_c"} or null, "move": {"from", "to", "count": <int> or "all"} or null}.
- Offsets are multiples of 10 minutes, from +0 up to (horizon - 10). Several steps may share an offset.
- "precool" runs the AC of a room (open or closed) toward target_c; "open" makes a room available for \
  seating with lights on; "move" transfers seated students; "close" switches a room off.
- Moves at the same offset run in the order listed. "count": "all" means every student still in the \
  source room at that point, after the earlier moves out of it. A move must fit the destination's free \
  seats (seats minus occupants, minus students moved in earlier at that offset); a move that does not fit \
  is rejected, not trimmed. To split a group that no single room can take, move an integer count first, \
  then "all" for the rest.
Prefer few, distinct, plausible plans. Return {"plans": []} if nothing staged would help."""


class LLMStagedProposer:
    name = "L2"

    def __init__(self, client: LLMClient, cfg: PlannerConfig, budget: CallBudget | None = None):
        self.client = client
        self.cfg = cfg
        self.budget = budget or CallBudget(cfg.llm_calls_per_step, cfg.llm_latency_s)

    def build_prompt(self, snap: Snapshot, fc: Forecast, feedback: list[dict[str, Any]]) -> str:
        H = self.cfg.horizon_steps
        payload = {
            "state": snap.summary(),
            "horizon_min": self.cfg.rollout_horizon_min,
            "demand_forecast_next_steps": [int(x) for x in fc.demand[:H]],
            "outdoor_forecast_c": [round(float(x), 2) for x in fc.t_out[: H + 1]],
            "forecast_adjustments": [
                {k: a[k] for k in ("source_id", "kind", "start", "end", "delta", "confidence") if k in a}
                for a in fc.adjustments
            ],
            "comfort_band_c": [self.cfg.band_low_c, self.cfg.band_high_c],
            "comfort_target_c": self.cfg.target_c,
            "precool_target_range_c": [PRECOOL_MIN_C, self.cfg.band_high_c],
            "valid_rooms": list(ROOM_IDS),
            "offset_step_min": STEP_MIN,
            "max_plans": self.cfg.l2_max_plans,
            "previous_rejections": feedback[:10],
            "example": {
                "plan_id": "p3",
                "steps": [
                    {"at": "+0", "open": ["B202"], "close": [], "precool": {"room": "B202", "target_c": 25.5}, "move": None},
                    {"at": "+20", "open": [], "close": [], "precool": None,
                     "move": {"from": "B201", "to": "B202", "count": "all"}},
                    {"at": "+20", "open": [], "close": ["B201"], "precool": None, "move": None},
                ],
                "rationale": "B201 AC offline; B202 idle, working AC, 45 seats",
            },
        }
        return "Current situation (all numbers come from the planner's tools):\n" + json.dumps(payload, indent=1)

    def propose(self, snap: Snapshot, fc: Forecast, feedback: list[dict[str, Any]]) -> ProposalBatch:
        user = self.build_prompt(snap, fc, feedback)
        r = call(self.client, self.budget.at_step(snap.step), role="L2", system=L2_SYSTEM, user=user, schema=L2_SCHEMA)
        tr = r.transcript()
        tr["step"] = snap.step
        if not r.ok:
            return ProposalBatch(macros=[], transcript=tr, error=r.error or "no data")
        plans = r.data.get("plans") if isinstance(r.data, dict) else None
        if not isinstance(plans, list):
            return ProposalBatch(macros=[], transcript=tr, error="schema: missing 'plans' list")
        return ProposalBatch(macros=plans[: self.cfg.l2_max_plans], transcript=tr)
