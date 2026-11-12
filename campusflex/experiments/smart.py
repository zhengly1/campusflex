"""Smart device extension experiment (IoT dispatch · AI usage profiles · DSA sensor control).

    python -m campusflex smart [--train-days 20] [--eval-days 10] [--workers N] [--llm]

The core planner (arm A, no LLM) decides the space plan of every evening; the smart layer
runs the six devices of each room on top of it. Training evenings (seeds 0-19, scenarios
steady/surge/heat/fault in turn) let the learning arms learn usage profiles and thermal
models online; evaluation evenings (seeds 100-109) inject two sensor faults each and freeze
what was learned.

Arms: ``static`` (fixed settings while a room is open), ``rule`` (occupancy on/off),
``smart_dsa_only`` (sensor-based fine control, default comfort settings),
``smart_profiles_only`` (learned profiles, raw sensor readings) and ``smart`` (both).

Outputs: ``results/smart/summary.json``, ``results/smart/profiles.json`` (the learned
memory of the ``smart`` arm; ``--profiles`` loads it as the starting point),
``results/smart/personas.md`` and ``results/smart/evening.json`` (every arm's two-minute
trace of one evaluation evening, ``--trace-day``, for the dashboard). All values are
simulated estimates.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ROOM_IDS, START_MIN, STEP_MIN
from ..smart.occupants import campus_habits
from ..smart.profiles import ATTRS, SLOTS, ProfileStore, narrate, slot_label
from ..smart.run import DayPlan, day_plan, identification_error, run_day
from ..smart.scheduler import RuleScheduler, SmartScheduler, StaticScheduler
from ..smart.sensors import SensorFault
from ..stats import paired_bootstrap
from .common import RESULTS, write_json

OUT = RESULTS / "smart"
SCENARIOS = ("steady", "surge", "heat", "fault")
TRAIN_SEEDS = tuple(range(0, 20))
EVAL_SEEDS = tuple(range(100, 110))
ARMS = ("static", "rule", "smart_dsa_only", "smart_profiles_only", "smart")
FAULT_KINDS = (("temp", "stuck", 0.0), ("co2", "drift", 6.0), ("lux", "spike", 400.0), ("co2", "stuck", 0.0),
               ("temp", "drift", 0.04))
TRACE_DAY = 0           # evaluation evening traced for the dashboard: seed 100, temperature stuck + CO2 drift


def eval_faults(plan: DayPlan, i: int) -> list[SensorFault]:
    """Two faults per evaluation evening, in rooms that are occupied when the fault starts."""
    out = []
    for j in range(2):
        sensor, kind, mag = FAULT_KINDS[(2 * i + j) % len(FAULT_KINDS)]
        start = 19 * 60 + 10 * ((3 * i + 5 * j) % 9)               # 19:00-20:20
        k = (start - START_MIN) // STEP_MIN
        used = [r for n, r in zip(plan.counts[k], ROOM_IDS) if n > 0] or list(ROOM_IDS)
        room = used[(i + 2 * j) % len(used)]
        out.append(SensorFault(room, sensor, kind, start, magnitude=mag))
    return out


def make_plans(train_days: int, eval_days: int, workers: int | None) -> tuple[list[DayPlan], list[DayPlan]]:
    specs = [(d, SCENARIOS[d % len(SCENARIOS)], s) for d, s in enumerate(TRAIN_SEEDS[:train_days])]
    specs += [(train_days + i, SCENARIOS[i % len(SCENARIOS)], s) for i, s in enumerate(EVAL_SEEDS[:eval_days])]
    with ProcessPoolExecutor(max_workers=workers or min(16, len(specs)), mp_context=_ctx()) as ex:
        plans = list(ex.map(_plan_job, specs))
    train, ev = plans[:train_days], plans[train_days:]
    for i, p in enumerate(ev):
        p.faults = eval_faults(p, i)
    return train, ev


def _ctx():
    """Workers start from a clean single-threaded server (``forkserver``): forking a process that
    runs MQTT or other threads could deadlock the children."""
    return mp.get_context("forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn")


def _plan_job(spec: tuple[int, str, int]) -> DayPlan:
    return day_plan(*spec)


def _scheduler(arm: str, store: ProfileStore | None):
    if arm == "static":
        return StaticScheduler()
    if arm == "rule":
        return RuleScheduler()
    if arm == "smart_dsa_only":
        return SmartScheduler(None, use_dsa=True)
    if arm == "smart_profiles_only":
        return SmartScheduler(store, use_dsa=False)
    return SmartScheduler(store, use_dsa=True)


def run_arm(args: tuple[str, list[DayPlan], list[DayPlan], dict[str, Any] | None, int]) -> dict[str, Any]:
    arm, train, ev, start_profiles, trace_day = args
    learns = arm in ("smart", "smart_profiles_only")
    store = (ProfileStore.from_dict(start_profiles) if start_profiles else ProfileStore()) if learns else None
    models: dict[str, Any] = {}
    train_rows = [run_day(p, _scheduler(arm, store), store=store, learn=learns, models=models) for p in train]
    ticks: list[dict[str, Any]] = []
    eval_rows = [run_day(p, _scheduler(arm, store), store=store, learn=False, models=models,
                         trace=ticks if i == trace_day else None) for i, p in enumerate(ev)]
    for r in train_rows + eval_rows:
        r.pop("adjustment_log", None)
    return {"arm": arm, "train": train_rows, "eval": eval_rows,
            "profiles": store.to_dict() if store is not None else None,
            "identification": identification_error(eval_rows[-1:]), "trace": ticks}


def profile_grid(store: ProfileStore) -> list[dict[str, Any]]:
    """What the learned memory recommends for every room and time slot, and where each value
    comes from (``room`` evidence, the campus-wide ``slot`` pool, or the ``default``)."""
    rows = []
    for room in ROOM_IDS:
        for k in range(len(SLOTS)):
            rec = store.recommend(room, k)
            rows.append({"room": room, "slot": slot_label(k), "setpoint": rec["setpoint"], "fan": rec["fan"],
                         "lux": round(rec["lux"]), "curtain": rec["curtain"], "projector": rec["projector"],
                         "projector_p": round(rec["projector_p"], 2), "source": rec["source"],
                         "evidence": rec["evidence"]})
    return rows


def evening(arms: dict[str, Any], ev: list[DayPlan], trace_day: int) -> dict[str, Any]:
    """The traced evaluation evening of every arm (same space plan, faults and occupants)."""
    p = ev[trace_day]
    keep = ("kwh", "kwh_by_device", "discomfort_pct", "adjustments", "adjustments_by_device",
            "scheduler_commands", "detection", "flags")
    return {"meta": {"eval_day_index": trace_day, "day": p.day, "scenario": p.scenario, "seed": p.seed,
                     "tick_min": 2, "note": "simulated estimates; per-room arrays in the order of rooms"},
            "rooms": list(ROOM_IDS), "faults": [f.to_dict() for f in p.faults],
            "arms": {a: {"metrics": {k: arms[a]["eval"][trace_day][k] for k in keep}, "ticks": arms[a]["trace"]}
                     for a in ARMS}}


def _mean(rows: list[dict[str, Any]], key: str, sub: str | None = None) -> float:
    vals = [(r[key][sub] if sub else r[key]) for r in rows]
    return float(np.mean(vals)) if vals else float("nan")


def arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "kwh": round(_mean(rows, "kwh"), 3),
        "kwh_by_device": {d: round(_mean(rows, "kwh_by_device", d), 3) for d in rows[0]["kwh_by_device"]},
        "discomfort_pct": {k: round(_mean(rows, "discomfort_pct", k), 3) for k in rows[0]["discomfort_pct"]},
        "adjustments_per_evening": round(_mean(rows, "adjustments"), 2),
        "adjustments_per_room_hour": round(_mean(rows, "adjustments_per_room_hour"), 4),
        "scheduler_commands_per_evening": round(_mean(rows, "scheduler_commands"), 1),
        "iot_missing_acks": int(sum(r["iot"]["missing_acks"] for r in rows)),
    }


def profile_accuracy(store: ProfileStore, habits: dict) -> dict[str, Any]:
    """Learned recommendation vs the latent habit, over (room, slot) cells with own evidence."""
    err: dict[str, list[float]] = {a: [] for a in ATTRS}
    cells = 0
    for (room, slot), h in habits.items():
        rec = store.recommend(room, slot)
        if rec["source"]["setpoint"] != "room" and rec["source"]["lux"] != "room":
            continue
        cells += 1
        err["setpoint"].append(abs(rec["setpoint"] - h["setpoint"]))
        err["fan"].append(float(rec["fan"] != h["fan"]))
        err["lux"].append(abs(rec["lux"] - h["lux"]))
        err["curtain"].append(abs(rec["curtain"] - h["curtain"]))
        err["projector"].append(float(rec["projector"] != (h["projector"] >= 0.5)))
    return {"cells_with_room_evidence": cells, "cells_total": len(habits),
            "setpoint_mae_c": _r(err["setpoint"]), "fan_mismatch_share": _r(err["fan"]),
            "lux_mae": _r(err["lux"]), "curtain_mae_pct": _r(err["curtain"]),
            "projector_mismatch_share": _r(err["projector"])}


def _r(v: list[float]) -> float | None:
    return round(float(np.mean(v)), 3) if v else None


def compare(a_rows: list[dict[str, Any]], b_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Paired by evaluation evening: a − b with a 95 % bootstrap interval."""
    out = {}
    for name, get in (("adjustments", lambda r: r["adjustments"]), ("kwh", lambda r: r["kwh"]),
                      ("thermal_discomfort_pct", lambda r: r["discomfort_pct"]["thermal"]),
                      ("light_discomfort_pct", lambda r: r["discomfort_pct"]["light"]),
                      ("co2_over_1000_pct", lambda r: r["discomfort_pct"]["co2"])):
        bs = paired_bootstrap([get(r) for r in a_rows], [get(r) for r in b_rows])
        out[name] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in bs.items()}
    return out


def run(train_days: int = len(TRAIN_SEEDS), eval_days: int = len(EVAL_SEEDS), workers: int | None = None,
        llm: bool = False, start_profiles: str | None = None, out_dir: Path = OUT,
        trace_day: int = TRACE_DAY) -> dict[str, Any]:
    t0 = time.time()
    train, ev = make_plans(train_days, eval_days, workers)
    trace_day = min(max(0, trace_day), len(ev) - 1)
    start = json.loads(Path(start_profiles).read_text()) if start_profiles else None
    with ProcessPoolExecutor(max_workers=len(ARMS), mp_context=_ctx()) as ex:
        arms = {r["arm"]: r for r in ex.map(run_arm, [(a, train, ev, start, trace_day) for a in ARMS])}
    habits = campus_habits()
    store = ProfileStore.from_dict(arms["smart"]["profiles"])
    smart_eval = arms["smart"]["eval"]
    detection = {k: sum((r["detection"][k] or 0) for r in smart_eval) for k in ("faults", "detected", "false_alarms")}
    detection["recall"] = round(detection["detected"] / detection["faults"], 3) if detection["faults"] else None
    flagged = detection["detected"] + detection["false_alarms"]
    detection["precision"] = round(detection["detected"] / flagged, 3) if flagged else None
    delays = [r["detection"]["mean_delay_min"] for r in smart_eval if r["detection"]["mean_delay_min"] is not None]
    detection["mean_delay_min"] = round(float(np.mean(delays)), 1) if delays else None
    client = None
    llm_note = None
    if llm:
        from .common import llm_client

        client, llm_note = llm_client(cache=True)
    text = narrate(store, client)
    res = {
        "meta": {"experiment": "smart device extension (IoT · AI · DSA)", "train_days": train_days,
                 "eval_days": eval_days, "train_seeds": list(TRAIN_SEEDS[:train_days]),
                 "eval_seeds": list(EVAL_SEEDS[:eval_days]), "scenarios": list(SCENARIOS),
                 "space_plan": "core planner arm A (masks only, no LLM)", "tick_min": 2,
                 "start_profiles": start_profiles, "seconds": round(time.time() - t0, 1),
                 "note": "simulated estimates on a synthetic campus; occupants' habits are synthetic"},
        "eval": {a: arm_summary(arms[a]["eval"]) for a in ARMS},
        "learning_curve_adjustments": {a: [r["adjustments"] for r in arms[a]["train"]] for a in ARMS},
        "comparisons": {"smart_vs_rule": compare(smart_eval, arms["rule"]["eval"]),
                        "smart_vs_static": compare(smart_eval, arms["static"]["eval"]),
                        "smart_vs_smart_dsa_only": compare(smart_eval, arms["smart_dsa_only"]["eval"]),
                        "smart_vs_smart_profiles_only": compare(smart_eval, arms["smart_profiles_only"]["eval"])},
        "profile_accuracy": profile_accuracy(store, habits),
        "anomaly_detection": detection,
        "faults_injected": [f for r in smart_eval for f in r["faults"]],
        "thermal_identification": arms["smart"]["identification"],
        "personas": text["personas"],
        "profile_grid": profile_grid(store),
        "persona_text": {"source": text["source"], "lines": text["lines"],
                         **({"llm_note": llm_note} if llm_note else {})},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", res)
    store.save(out_dir / "profiles.json")
    (out_dir / "evening.json").write_text(json.dumps(evening(arms, ev, trace_day), separators=(",", ":")))
    (out_dir / "personas.md").write_text(
        "# Learned usage personas per time slot\n\n"
        f"Learned over {train_days} simulated evenings from {store.n_adjustments} manual adjustments "
        f"({text['source']} text; simulated data).\n\n" + "\n".join(f"- {ln}" for ln in text["lines"]) + "\n")
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-days", type=int, default=len(TRAIN_SEEDS))
    ap.add_argument("--eval-days", type=int, default=len(EVAL_SEEDS))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--llm", action="store_true", help="describe the personas with the LLM (numbers checked)")
    ap.add_argument("--profiles", default=None, help="start from a saved profiles.json instead of an empty memory")
    ap.add_argument("--out-dir", default=str(OUT),
                    help="where to write summary.json, profiles.json, personas.md and evening.json")
    ap.add_argument("--trace-day", type=int, default=TRACE_DAY,
                    help="index of the evaluation evening traced in evening.json (default %(default)s)")
    a = ap.parse_args(argv)
    res = run(a.train_days, a.eval_days, a.workers, a.llm, a.profiles, Path(a.out_dir), a.trace_day)
    print(json.dumps({k: res[k] for k in ("eval", "profile_accuracy", "anomaly_detection", "thermal_identification")},
                     indent=1))


if __name__ == "__main__":
    main()
