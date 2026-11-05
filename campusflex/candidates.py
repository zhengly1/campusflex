"""Candidate generators: the 64 v1 room masks and the rule-staged generator (Section 3.3)."""
from __future__ import annotations

from itertools import combinations
from typing import Any

from .config import ROOM_IDS, PlannerConfig
from .forecast import Forecast
from .plans import CompiledPlan, mask_plan
from .state import Snapshot


def all_masks() -> list[frozenset[str]]:
    """All 2^6 = 64 subsets of rooms (including the empty mask)."""
    out = []
    for k in range(len(ROOM_IDS) + 1):
        out += [frozenset(c) for c in combinations(ROOM_IDS, k)]
    return out


ALL_MASKS = all_masks()


def mask_candidates(snap: Snapshot, cfg: PlannerConfig, fill: str = "smart") -> list[CompiledPlan]:
    plans = []
    for m in ALL_MASKS:
        p = mask_plan(m, snap.health, cfg, cfg.horizon_steps, snap.step)
        if fill != "smart":
            from dataclasses import replace

            p.actions = [replace(a, fill=fill) for a in p.actions]
        plans.append(p)
    return plans


def rule_staged_macros(snap: Snapshot, fc: Forecast, cfg: PlannerConfig) -> list[dict[str, Any]]:
    """Rule-staged generator (non-LLM control for L2).

    For every room that is offline, above ``rule_staged_temp_c``, or forecast to exceed
    capacity, pair it with each idle room that has enough seats and working AC, and emit
    pre-cool durations from ``precool_set_min`` followed by a full transfer.
    """
    macros: list[dict[str, Any]] = []
    idle = [
        i
        for i, r in enumerate(ROOM_IDS)
        if not snap.open[i] and snap.capacity[i] > 0 and snap.health[i] > 0
    ]
    durations = [d for d in cfg.precool_set_min if 0 < d < cfg.rollout_horizon_min]
    target = round(cfg.target_c - 0.5, 1)
    sources = [
        i
        for i, r in enumerate(ROOM_IDS)
        if snap.counts[i] > 0 and (snap.health[i] <= 0 or snap.temps[i] > cfg.rule_staged_temp_c)
    ]
    for i in sources:
        for j in idle:
            if snap.capacity[j] < snap.counts[i]:
                continue
            for d in durations:
                macros.append(
                    {
                        "plan_id": f"r:{ROOM_IDS[i]}>{ROOM_IDS[j]}@{d}",
                        "steps": [
                            {"at": "+0", "open": [], "close": [], "move": None,
                             "precool": {"room": ROOM_IDS[j], "target_c": target}},
                            {"at": f"+{d}", "open": [ROOM_IDS[j]], "close": [], "precool": None,
                             "move": {"from": ROOM_IDS[i], "to": ROOM_IDS[j], "count": "all"}},
                            {"at": f"+{d}", "open": [], "close": [ROOM_IDS[i]], "precool": None, "move": None},
                        ],
                        "rationale": "rule: transfer out of offline/warm room after pre-cooling",
                    }
                )
    # Forecast to exceed open capacity: pre-cool an idle room, then open it.
    open_cap = int(sum(snap.capacity[i] for i in range(len(ROOM_IDS)) if snap.open[i]))
    peak = int(max(fc.demand[: cfg.horizon_steps])) if len(fc.demand) else snap.demand
    if peak > open_cap:
        for j in idle:
            for d in durations:
                macros.append(
                    {
                        "plan_id": f"r:open{ROOM_IDS[j]}@{d}",
                        "steps": [
                            {"at": "+0", "open": [], "close": [], "move": None,
                             "precool": {"room": ROOM_IDS[j], "target_c": target}},
                            {"at": f"+{d}", "open": [ROOM_IDS[j]], "close": [], "precool": None, "move": None},
                        ],
                        "rationale": "rule: forecast demand exceeds open capacity",
                    }
                )
    return macros
