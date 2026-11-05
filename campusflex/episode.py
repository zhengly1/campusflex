"""Episode runner: observe → decide → commit first action → advance time (Figure 2).

This is the fast, direct loop used for batch experiments. ``campusflex.agent.graph`` runs
the same nodes as a LangGraph state graph.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import PlannerConfig
from .devices import DeviceLayer
from .env import CampusEnv
from .metrics import episode_metrics
from .policies import Decision, hold_action
from .scenarios import Scenario


@dataclass
class EpisodeResult:
    scenario: str
    seed: int
    policy: str
    mismatch: bool
    metrics: dict[str, Any]
    steps: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)
    policy_stats: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "seed": self.seed,
            "policy": self.policy,
            "mismatch": self.mismatch,
            **self.metrics,
            **{f"llm_{k}": v for k, v in self.policy_stats.items()},
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(
                {"summary": self.summary(), "steps": self.steps, "decisions": self.decisions, "counters": self.counters},
                indent=1,
                default=str,
            )
        )


def decision_record(dec: Decision) -> dict[str, Any]:
    return {
        "plan_id": dec.plan_id,
        "source": dec.source,
        "trigger": dec.trigger,
        "ranking": dec.ranking,
        "n_candidates": dec.n_candidates,
        "n_rejected": len(dec.rejected),
        "rejected_sample": dec.rejected[:5] if dec.source != "baseline" else [],
        "llm": dec.llm,
        "forecast": dec.forecast,
        "notes": dec.notes,
        "macro": dec.plan.macro if dec.plan else None,
    }


def commit(env: CampusEnv, snap, dec: Decision, key: str, cfg: PlannerConfig) -> dict[str, Any]:
    """Commit node: apply the first action; on rejection fall back to holding the current state."""
    res = env.apply_plan(dec.action, snap.version, key, {"plan_id": dec.plan_id, "source": dec.source})
    out = {"committed": dec.plan_id, "ack": res.executed, "outcomes": res.outcomes, "rejected": res.rejected}
    if not res.executed and not res.duplicate:
        # A stale snapshot (the belief changed after observation) holds from the current state.
        if any(v.get("code") == "stale_snapshot" for v in res.rejected):
            snap = env.get_state()
        hold = hold_action(snap, cfg, fill=dec.action.fill)
        r2 = env.apply_plan(hold, snap.version, key + ":hold", {"plan_id": "hold", "source": "fallback"})
        out.update({"committed": "hold", "ack": r2.executed, "fallback": True})
    return out


def run_episode(
    scenario: Scenario,
    policy: Any,
    cfg: PlannerConfig,
    device_layer: DeviceLayer | None = None,
    keep_decisions: bool = False,
    on_step: Any = None,
) -> EpisodeResult:
    env = CampusEnv(scenario, cfg, device_layer)
    decisions = []
    try:                                   # an MQTT layer's network thread stops even if a step raises
        policy.reset()
        snap = env.get_state()
        while snap is not None:
            dec = policy.decide(snap)
            key = f"{scenario.name}/s{scenario.seed}/{policy.name}/t{snap.step}"
            c = commit(env, snap, dec, key, cfg)
            if "recheck" in (dec.notes or {}):
                for r in dec.notes["recheck"]:
                    env.request_recheck(r)
            # A committed fallback (hold) is not the plan that was predicted: no prediction to compare.
            p = None if c.get("fallback") else dec.predicted
            extra = {
                "plan_id": c["committed"] if c.get("fallback") else dec.plan_id,
                "source": "fallback" if c.get("fallback") else dec.source,
                "ack": c["ack"],
                "pred_first_V": round(p.first_V, 6) if p else None,
                "pred_first_E": round(p.first_E, 6) if p else None,
                "pred_V": round(p.V, 6) if p else None,
                "pred_E": round(p.E, 6) if p else None,
                "pred_omega": round(p.Omega, 6) if p else None,
                "trigger": dec.trigger,
            }
            if keep_decisions:
                decisions.append({"step": snap.step, "state_version": snap.version, **decision_record(dec), **c})
            if on_step is not None:
                on_step(env, snap, dec, c)
            snap = env.advance_time(extra)
    finally:
        env.device.close()
    return EpisodeResult(
        scenario=scenario.name,
        seed=scenario.seed,
        policy=policy.name,
        mismatch=scenario.mismatch is not None,
        metrics=episode_metrics(env.step_logs, cfg),
        steps=env.step_logs,
        decisions=decisions,
        counters=dict(env.counters),
        policy_stats=dict(getattr(policy, "stats", {}) or {}),
    )
