"""Smart devices page of the CampusFlex dashboard: per-device IoT dispatch, learned usage
profiles (AI) and sensor-based fine control (DSA), from ``python -m campusflex smart``.

    streamlit run campusflex/dashboard/app.py -- --smart-dir results/smart

Reads ``summary.json`` and ``evening.json`` of a smart results directory (``--smart-dir`` or
the sidebar). As on the decision-trace page, every label is the log field a number came from.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
DISCOMFORT = ("thermal", "light", "co2", "glare")
ROOM_FIELDS = ("occupancy", "temp_true", "temp_est", "co2_est", "lux_est", "setpoint", "fan", "light", "vent",
               "curtain", "projector", "sockets", "kw", "fault_active", "suspect")


def _wide_table(df: Any) -> None:
    try:
        st.dataframe(df, width="stretch")
    except Exception:  # noqa: BLE001 - older Streamlit releases want use_container_width
        st.dataframe(df, use_container_width=True)


def _arg_dir() -> str:
    argv = sys.argv[1:]
    if "--smart-dir" in argv:
        i = argv.index("--smart-dir")
        if i + 1 < len(argv):
            return argv[i + 1]
    return str(ROOT / "results" / "smart")


@st.cache_data
def _load(path: str, mtime: float) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _json(path: Path) -> dict[str, Any] | None:
    return _load(str(path), path.stat().st_mtime) if path.exists() else None


def _metrics(prefix: str, d: dict[str, Any], keys: tuple[str, ...]) -> None:
    keys = tuple(k for k in keys if k in d)
    for col, k in zip(st.columns(max(1, len(keys))), keys, strict=True):
        col.metric(f"{prefix}.{k}", d[k])


def evaluation(summ: dict[str, Any]) -> None:
    m = summ.get("meta", {})
    st.caption(f"{m.get('train_days')} training and {m.get('eval_days')} evaluation evenings (two injected sensor "
               f"faults each); space plan: {m.get('space_plan')}. {m.get('note', '')}")
    st.subheader("evaluation evenings, mean per evening (summary.eval)")
    ev = summ["eval"]
    _wide_table(pd.DataFrame.from_dict(
        {a: {"kwh": s["kwh"], **{f"discomfort_pct.{k}": s["discomfort_pct"][k] for k in DISCOMFORT},
             "adjustments_per_evening": s["adjustments_per_evening"],
             "scheduler_commands_per_evening": s["scheduler_commands_per_evening"],
             "iot_missing_acks": s["iot_missing_acks"]} for a, s in ev.items()}, orient="index"))
    st.caption("discomfort_pct = share of occupied person-minutes; adjustments = manual wall-panel commands.")
    st.subheader("summary.eval[*].kwh_by_device")
    st.bar_chart(pd.DataFrame({a: s["kwh_by_device"] for a, s in ev.items()}).T)
    rows = [{"comparison": name, "metric": metric, **{k: v.get(k) for k in ("mean_diff", "lo", "hi", "excludes_zero")}}
            for name, c in summ.get("comparisons", {}).items() for metric, v in c.items()]
    if rows:
        st.subheader("paired by evaluation evening, smart − other, 95 % bootstrap (summary.comparisons)")
        _wide_table(pd.DataFrame(rows))


def profiles(summ: dict[str, Any]) -> None:
    st.header("AI · usage profile per time slot")
    lc = summ.get("learning_curve_adjustments")
    if lc:
        st.subheader("manual adjustments per training evening (summary.learning_curve_adjustments)")
        st.line_chart(pd.DataFrame(lc).rename_axis("training evening"))
    if summ.get("personas"):
        st.subheader("campus-wide persona per slot (summary.personas)")
        _wide_table(pd.DataFrame([{"slot": p["slot"], "tags": ", ".join(p["tags"]), **p["values"],
                                   "evidence": p["evidence"],
                                   "rooms_that_differ": "; ".join(
                                       " ".join(f"{k} {v}" for k, v in r.items()) for r in p["rooms_that_differ"])}
                                  for p in summ["personas"]]).set_index("slot"))
        pt = summ.get("persona_text", {})
        with st.expander(f"summary.persona_text.lines ({pt.get('source')})"):
            for line in pt.get("lines", []):
                st.markdown(f"- {line}")
    grid = summ.get("profile_grid")
    if grid:
        st.subheader("what the memory recommends per room and slot (summary.profile_grid)")
        field = st.selectbox("profile_grid field", ("setpoint", "fan", "lux", "curtain", "projector_p"))
        g = pd.DataFrame(grid)
        _wide_table(g.pivot(index="room", columns="slot", values=field))
        src = g.assign(src=[r["source"]["projector" if field == "projector_p" else field] for r in grid])
        st.caption("profile_grid[*].source – room: the room's own evidence; slot: campus-wide pool for the slot; "
                   "default: no evidence yet")
        _wide_table(src.pivot(index="room", columns="slot", values="src"))
    if summ.get("profile_accuracy"):
        st.subheader("learned vs latent habits (summary.profile_accuracy)")
        _metrics("profile_accuracy", summ["profile_accuracy"],
                 ("cells_with_room_evidence", "setpoint_mae_c", "lux_mae", "curtain_mae_pct", "fan_mismatch_share",
                  "projector_mismatch_share"))


def sensing(summ: dict[str, Any]) -> None:
    st.header("DSA · sensor fusion and fault detection")
    if summ.get("anomaly_detection"):
        st.subheader("stuck / drift faults, smart arm (summary.anomaly_detection)")
        _metrics("anomaly_detection", summ["anomaly_detection"],
                 ("faults", "detected", "false_alarms", "recall", "precision", "mean_delay_min"))
    ti = summ.get("thermal_identification", {})
    if ti:
        st.subheader("identified thermal model vs Table 6 (summary.thermal_identification)")
        cols = st.columns(2)
        for col, k in zip(cols, ("R", "C"), strict=True):
            col.metric(f"thermal_identification.{k}.median_rel_error", (ti.get(k) or {}).get("median_rel_error"))
    if summ.get("faults_injected"):
        with st.expander(f"summary.faults_injected ({len(summ['faults_injected'])})"):
            _wide_table(pd.DataFrame(summ["faults_injected"]))


def _clock(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}"


def evening(evg: dict[str, Any]) -> None:
    m, rooms, arms = evg["meta"], evg["rooms"], evg["arms"]
    st.header(f"IoT · one evaluation evening: seed {m['seed']}, scenario {m['scenario']} (evening.json)")
    st.caption("Same space plan, occupants and sensor faults for every arm; one record every "
               f"{m.get('tick_min', 2)} minutes. The fault detectors run in every arm, but only the DSA arms "
               "(smart, smart_dsa_only) control from the fused estimates.")
    faults = pd.DataFrame(evg["faults"])
    if not faults.empty:
        faults.insert(0, "start", faults["start_min"].map(_clock))
        st.write("evening.faults")
        _wide_table(faults)
    _wide_table(pd.DataFrame.from_dict(
        {a: {"kwh": x["metrics"]["kwh"], "adjustments": x["metrics"]["adjustments"],
             "scheduler_commands": x["metrics"]["scheduler_commands"],
             **{f"discomfort_pct.{k}": x["metrics"]["discomfort_pct"][k] for k in DISCOMFORT},
             "detection.detected": x["metrics"]["detection"]["detected"],
             "detection.false_alarms": x["metrics"]["detection"]["false_alarms"]} for a, x in arms.items()},
        orient="index"))
    clocks = [t["clock"] for t in next(iter(arms.values()))["ticks"]]
    st.subheader("manual adjustments so far, by arm (ticks[*].events, by = user)")
    st.line_chart(pd.DataFrame({a: pd.Series([sum(e["by"] == "user" and e["ok"] for e in t["events"])
                                              for t in x["ticks"]]).cumsum().tolist() for a, x in arms.items()},
                               index=clocks))

    names = list(arms)
    c1, c2 = st.columns(2)
    arm = c1.selectbox("arm", names, index=names.index("smart") if "smart" in names else 0)
    faulted = [f["room"] for f in evg["faults"]]
    room = c2.selectbox("room", rooms, index=rooms.index(faulted[0]) if faulted else 0)
    i, ticks = rooms.index(room), arms[arm]["ticks"]
    df = pd.DataFrame({f: [t[f][i] for t in ticks] for f in ("occupancy", "temp_true", "temp_sensor", "temp_est",
                                                             "setpoint", "co2_true", "co2_sensor", "co2_est",
                                                             "lux_true", "lux_sensor", "lux_est", "fan", "vent",
                                                             "light", "curtain", "projector", "sockets", "kw")},
                      index=clocks).astype(float)
    p = f"ticks[*].{{field}}[{room}]"
    a, b = st.columns(2)
    a.write(p.format(field="temp_* / setpoint") + " (°C)")
    a.line_chart(df[["temp_true", "temp_sensor", "temp_est", "setpoint"]])
    b.write(p.format(field="co2_*") + " (ppm)")
    b.line_chart(df[["co2_true", "co2_sensor", "co2_est"]])
    a.write(p.format(field="lux_*"))
    a.line_chart(df[["lux_true", "lux_sensor", "lux_est"]])
    b.write(p.format(field="light / curtain") + " (%)")
    b.line_chart(df[["light", "curtain"]])
    a.write(p.format(field="fan / vent / projector / sockets") + " (level)")
    a.line_chart(df[["fan", "vent", "projector", "sockets"]])
    b.write(p.format(field="occupancy"))
    b.area_chart(df[["occupancy"]])

    flagged = [(t["clock"], f) for t in ticks for f in t["flags"] if f["room"] == room]
    for clock, f in flagged:
        st.warning(f"{clock} ticks[*].flags: {f['sensor']} sensor {f['kind']} – its readings are dropped from "
                   "the estimate")
    ev = [{"clock": t["clock"], "by": e["by"], "device": e["device"], "value": json.dumps(e["value"]),
           "ok": e["ok"], "reason": e["reason"]} for t in ticks for e in t["events"] if e["room"] == room]
    st.subheader(f"commands to {room} under {arm} (ticks[*].events)")
    if ev:
        _wide_table(pd.DataFrame(ev))
    else:
        st.caption("no commands")

    clock = st.select_slider("tick (ticks[*].clock)", options=clocks, value=clocks[min(45, len(clocks) - 1)])
    t = ticks[clocks.index(clock)]
    st.subheader(f"all rooms at {clock} under {arm}")
    st.caption(f"ticks[{clocks.index(clock)}]: slot {t['slot']}, t_out {t['t_out']} °C")
    _wide_table(pd.DataFrame({f: [json.dumps(v) if isinstance(v, dict) else v for v in t[f]] for f in ROOM_FIELDS},
                             index=rooms))
    if t["events"]:
        _wide_table(pd.DataFrame([{**e, "value": json.dumps(e["value"])} for e in t["events"]]))


def render() -> None:
    st.title("CampusFlex – smart devices")
    st.caption("Per-device IoT dispatch, learned usage profiles (AI) and sensor-based control (DSA). Simulated "
               "estimates on a synthetic campus with synthetic occupants. Labels are the log field names.")
    d = Path(st.sidebar.text_input("Smart results dir", _arg_dir()))
    summ = _json(d / "summary.json")
    if summ is None or "eval" not in summ:
        st.warning(f"{d / 'summary.json'} not found. Create it with `python -m campusflex smart --out-dir {d}`.")
        return
    tabs = st.tabs(["evaluation", "AI · profiles", "DSA · sensors", "IoT · one evening"])
    with tabs[0]:
        evaluation(summ)
    with tabs[1]:
        profiles(summ)
    with tabs[2]:
        sensing(summ)
    with tabs[3]:
        evg = _json(d / "evening.json")
        if evg is None:
            st.info(f"{d / 'evening.json'} not found: rerun `python -m campusflex smart` to trace one evening.")
        else:
            evening(evg)


render()
