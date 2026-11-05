"""Integrated case trace T1 (Section 4.5).

Notice-plus-fault scenario on held-out seed 10. At 19:20 notice N1 announces a review session
ending 19:40 with about 60 students; at 20:00 the device layer reports B201's AC offline.
Each decision step writes a record with the inputs (notice → L3 delta, IoT message, L2 plan,
simulation), the top-three ranking, the committed action and three counterfactual replays
that restart from the same saved state with one input removed or delayed:
``no_notice``, ``fault_msg_delayed_one_step`` and ``no_L2``.

T1 is complete when one committed action can be traced to all four inputs and removing any
one of them changes it. If none does, the report says the inputs were present but not
jointly decisive.

The live run needs an LLM (L3 + L2). ``--rehearsal`` runs the identical pipeline with the
non-LLM substitutes (keyword parser for L3, rule-staged generator for L2) and writes
``results/trace_T1_rehearsal.json`` – it is labelled as such and is not ledger item 6.
"""
from __future__ import annotations

import argparse
import copy
import json
from typing import Any

import numpy as np

from ..config import ROOM_INDEX, PlannerConfig, clock_str
from ..env import CampusEnv
from ..episode import commit
from ..forecast import ConstantForecaster, NoticeForecaster
from ..llm.client import PROVIDER_MODELS
from ..policies import Decision, SimulationPlanner
from ..scenarios import make_scenario
from ..state import Snapshot
from .common import write_json

FAULT_ROOM = "B201"
NOTICE_ID = "N1"


def _own_budget(obj: Any, budgets: dict[int, Any] | None = None) -> Any:
    """Counterfactual replays get their own LLM call budget so they never starve the real decision.

    Roles that share one budget in the real planner (L3 and L2) share one new budget in the
    clone (``budgets`` maps the original budget to its replacement). Per-instance logs
    (transcripts, errors, screened ids) are fresh lists, so a replay never writes into the real
    role's records."""
    if obj is None or not hasattr(obj, "budget"):
        return obj
    from ..llm.client import CallBudget

    o = copy.copy(obj)
    b = obj.budget
    if b is None:
        o.budget = None
    else:
        budgets = {} if budgets is None else budgets
        if id(b) not in budgets:
            budgets[id(b)] = CallBudget(b.max_calls, b.latency_s)
        o.budget = budgets[id(b)]
    for attr in ("transcripts", "errors", "screened", "log"):
        if isinstance(getattr(o, attr, None), list):
            setattr(o, attr, [])
    return o


def _clone_forecaster(f: Any, drop: set[str], budgets: dict[int, Any] | None = None) -> Any:
    if isinstance(f, NoticeForecaster):
        g = NoticeForecaster(_own_budget(f.interpreter, budgets), f.cfg)
        g._seen = set(f._seen)
        g._adjs = [copy.deepcopy(a) for a in f._adjs if str(a.get("source_id")) not in drop]
        return g
    return ConstantForecaster()


def clone_planner(p: SimulationPlanner, *, l2: Any = "keep", use_rule: bool | None = None,
                  drop_sources: set[str] = frozenset()) -> SimulationPlanner:
    budgets: dict[int, Any] = {}
    q = SimulationPlanner(
        p.cfg,
        forecaster=_clone_forecaster(p.forecaster, set(drop_sources), budgets),
        use_masks=p.use_masks,
        use_rule=p.use_rule if use_rule is None else use_rule,
        l2=_own_budget(p.l2, budgets) if l2 == "keep" else l2,
        name=p.name + "/cf",
    )
    q.carry = copy.deepcopy(p.carry)
    q.prev_health = None if p.prev_health is None else p.prev_health.copy()
    q.prev_demand = p.prev_demand
    q.l2_feedback = list(p.l2_feedback)
    return q


def _snap_without_notice(s: Snapshot, nid: str) -> Snapshot:
    t = copy.copy(s)
    t.notices = [n for n in s.notices if n.notice_id != nid]
    return t


def _snap_fault_delayed(s: Snapshot, prev_health: np.ndarray | None) -> Snapshot:
    t = copy.copy(s)
    i = ROOM_INDEX[FAULT_ROOM]
    if prev_health is not None and s.health[i] <= 0 < prev_health[i]:
        t.health = s.health.copy()
        t.health[i] = prev_health[i]
    return t


def _staged_source(rehearsal: bool) -> str:
    return "rule" if rehearsal else "L2"


def run(planner: SimulationPlanner, cfg: PlannerConfig, device_kind: str = "mqtt", seed: int = 10,
        rehearsal: bool = False, out: str | None = None) -> dict[str, Any]:
    from ..iot import make_device_layer

    sc = make_scenario("notice_fault", seed)
    layer = make_device_layer(device_kind)
    env = CampusEnv(sc, cfg, layer)
    try:                                   # stop the MQTT network thread even if a step raises
        planner.reset()
        records: list[dict[str, Any]] = []
        snap = env.get_state()
        prev_health = None
        staged = _staged_source(rehearsal)
        while snap is not None:
            # counterfactuals from the same saved state (before the actual decision mutates the planner)
            cf: dict[str, str] = {}
            if snap.clock_min >= 19 * 60 + 20:
                d1 = clone_planner(planner, drop_sources={NOTICE_ID}).decide(_snap_without_notice(snap, NOTICE_ID))
                d2 = clone_planner(planner).decide(_snap_fault_delayed(snap, prev_health))
                d3 = (clone_planner(planner, use_rule=False) if rehearsal else clone_planner(planner, l2=None)).decide(snap)
                cf = {"no_notice": d1.plan_id, "fault_msg_delayed_one_step": d2.plan_id,
                      f"no_{staged}": d3.plan_id}
            dec: Decision = planner.decide(snap)
            c = commit(env, snap, dec, f"T1/t{snap.step}", cfg)
            tel = env.device.latest().get(FAULT_ROOM)
            n1 = [l for l in getattr(planner.forecaster, "log", []) if str((l.get("raw") or {}).get("source_id")) == NOTICE_ID]
            applied = next((l["applied"] for l in reversed(n1) if l.get("applied")), None)
            top_staged = next((r for r in dec.ranking if r["source"].endswith(staged)), None)
            p = dec.predicted
            rec = {
                "step": clock_str(snap.clock_min),
                "state_version": snap.version,
                "inputs": {
                    "notice": {"id": NOTICE_ID, "l3_delta": None if applied is None else applied["delta"],
                               "confidence": None if applied is None else applied["confidence"]},
                    "iot": {"msg_id": f"campus/{FAULT_ROOM}/telemetry/{tel.seq}" if tel else None,
                            "health": int(tel.health > 0) if tel else None,
                            "ts": clock_str(tel.ts_min) if tel else None,
                            "ack": c["outcomes"].get(FAULT_ROOM) == "ack" if c.get("outcomes") else c["ack"]},
                    "llm_plan": {"plan_id": top_staged["plan_id"] if top_staged else None, "source": staged},
                    "sim": {"pred_V": round(p.V, 4) if p else None, "pred_E": round(p.E, 4) if p else None,
                            "omega": round(p.Omega, 4) if p else None},
                },
                "ranking": [{"plan_id": r["plan_id"], "source": r["source"], "score_J": r["score_J"]} for r in dec.ranking[:3]],
                "committed": {"plan_id": dec.plan_id, "source": dec.source, "ack": c["ack"]},
                "counterfactuals": cf,
            }
            records.append(rec)
            prev_health = snap.health.copy()
            snap = env.advance_time({"plan_id": dec.plan_id, "source": dec.source,
                                     "pred_first_V": round(p.first_V, 6) if p else None})
    finally:
        env.device.close()

    def decisive(r: dict[str, Any]) -> bool:
        cfs = r["counterfactuals"]
        return (bool(cfs) and r["committed"]["source"].endswith(staged)
                and r["inputs"]["notice"]["l3_delta"] is not None
                and all(v != r["committed"]["plan_id"] for v in cfs.values()))

    joint = [r["step"] for r in records if decisive(r)]
    from .common import config_hash, frozen_config, freeze_meta, llm_freeze_meta, notices_freeze_meta

    _, fh, ffrozen = frozen_config()
    h = config_hash(cfg)
    client = getattr(getattr(planner, "l2", None), "client", None)
    res = {
        "meta": {"experiment": "trace T1" + (" (REHEARSAL: non-LLM substitutes)" if rehearsal else ""),
                 **freeze_meta(h, ffrozen and h == fh), **notices_freeze_meta(),
                 **(llm_freeze_meta(client) if client is not None and not rehearsal else {}),
                 "scenario": "notice_fault", "seed": seed, "device_layer": getattr(layer, "name", device_kind),
                 "device_fallback": bool(getattr(layer, "fallback", False)),
                 "device_fallback_reason": getattr(layer, "fallback_reason", None),
                 "planner": planner.name, "forecaster": getattr(planner.forecaster, "name", "?")},
        "complete": bool(joint),
        "jointly_decisive_steps": joint,
        "statement": ("committed action traced to notice, IoT message, staged plan and simulation; removing any "
                      "one input changes it" if joint else
                      "inputs were present but not jointly decisive at any step"),
        "records": records,
    }
    write_json(out or ("results/trace_T1_rehearsal.json" if rehearsal else "results/trace_T1.json"), res)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Integrated case trace T1")
    ap.add_argument("--rehearsal", action="store_true", help="non-LLM substitutes (keyword L3, rule-staged L2)")
    ap.add_argument("--device", default="mqtt", choices=["mqtt", "bus", "direct"])
    ap.add_argument("--provider", default=None, choices=list(PROVIDER_MODELS),
                    help="LLM provider (default: CAMPUSFLEX_PROVIDER or deepseek)")
    ap.add_argument("--model", default=None, help="model id (default: provider default)")
    a = ap.parse_args()
    from .common import frozen_config

    cfg = frozen_config()[0]                 # held-out seed 10: the configuration locked at the freeze gate
    if a.rehearsal:
        from ..roles.l3_events import KeywordParser

        planner = SimulationPlanner(cfg, forecaster=NoticeForecaster(KeywordParser(), cfg), use_rule=True, name="T1-rehearsal")
    else:
        from ..llm.client import CallBudget, live_client_or_reason
        from ..roles.l2_proposer import LLMStagedProposer
        from ..roles.l3_events import LLMInterpreter

        client, reason = live_client_or_reason(cache=True, kind=a.provider, model=a.model)
        if client is None:
            write_json("results/trace_T1.json", {"meta": {"experiment": "trace T1"}, "status": "not_run", "reason": reason})
            print("T1 not run:", reason, "(use --rehearsal for the non-LLM pipeline check)")
            return
        budget = CallBudget(cfg.llm_calls_per_step, cfg.llm_latency_s)
        planner = SimulationPlanner(cfg, forecaster=NoticeForecaster(LLMInterpreter(client, cfg, budget), cfg),
                                    use_rule=True, l2=LLMStagedProposer(client, cfg, budget), name="T1-D")
    res = run(planner, cfg, a.device, rehearsal=a.rehearsal)
    print(json.dumps({k: res[k] for k in ("meta", "complete", "jointly_decisive_steps", "statement")}, indent=1))


if __name__ == "__main__":
    main()
