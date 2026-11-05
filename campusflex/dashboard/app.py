"""CampusFlex dashboard (Table 11): rooms, temperatures, assignments, candidate ranking with
source (mask, rule, L2) and the L4 dialogue.

    streamlit run campusflex/dashboard/app.py -- --episode results/demo_episode.json

Every number on screen is read straight from a log field of the episode JSON written by
``EpisodeResult.save``; each table column / metric label is the field name it came from.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
ROOM_IDS = ("A101", "A102", "A103", "B201", "B202", "B203")   # order of per-room log arrays


def _wide_table(df: Any) -> None:
    """Full-width table on any supported Streamlit: ``width="stretch"`` (newer releases) or
    ``use_container_width`` (older releases, where ``width`` must be an integer)."""
    try:
        st.dataframe(df, width="stretch")
    except Exception:  # noqa: BLE001 - StreamlitAPIException / TypeError on older releases
        st.dataframe(df, use_container_width=True)


def _arg_episode() -> str:
    argv = sys.argv[1:]
    if "--episode" in argv:
        i = argv.index("--episode")
        if i + 1 < len(argv):
            return argv[i + 1]
    return str(ROOT / "results" / "demo_episode.json")


@st.cache_data
def load_episode(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def per_room_frame(steps: list[dict[str, Any]], field: str) -> pd.DataFrame:
    rows = {s["clock"]: s[field] for s in steps if isinstance(s.get(field), list)}
    return pd.DataFrame.from_dict(rows, orient="index", columns=list(ROOM_IDS))


def main() -> None:
    st.set_page_config(page_title="CampusFlex dashboard", layout="wide")
    st.title("CampusFlex – decision trace")
    st.caption("Simulated estimates from a synthetic six-room campus. Labels are the log field names.")
    path = st.sidebar.text_input("Episode JSON", _arg_episode())
    if not Path(path).exists():
        st.warning(f"{path} not found. Create it with `python -m campusflex.dashboard.export_demo`.")
        return
    ep = load_episode(path)
    summ, steps, decs = ep.get("summary", {}), ep.get("steps", []), ep.get("decisions", [])

    st.subheader("summary")
    keys = [k for k in ("scenario", "seed", "policy", "kwh", "service_pct", "dev_c", "moves", "switches", "score",
                        "unserved_person_min", "hard_violations") if k in summ]
    cols = st.columns(min(6, max(1, len(keys))))
    for i, k in enumerate(keys):
        cols[i % len(cols)].metric(f"summary.{k}", summ[k])
    if ep.get("counters"):
        st.json({"counters": ep["counters"]}, expanded=False)

    if not steps:
        st.info("episode has no step logs")
        return
    st.subheader("steps[*].T_end (°C) by room")
    st.line_chart(per_room_frame(steps, "T_end"))
    st.subheader("steps[*].counts (students) by room")
    st.area_chart(per_room_frame(steps, "counts"))

    clocks = [s["clock"] for s in steps]
    clock = st.select_slider("step (steps[*].clock)", options=clocks, value=clocks[min(12, len(clocks) - 1)])
    k = clocks.index(clock)
    s = steps[k]
    st.subheader(f"rooms at {clock}")
    room_df = pd.DataFrame(
        {f: s.get(f) for f in ("open", "counts", "T_start", "T_end", "u", "health") if isinstance(s.get(f), list)},
        index=list(ROOM_IDS),
    )
    _wide_table(room_df)
    scal = {f"steps[{k}].{f}": s[f] for f in ("demand", "served", "unserved", "E", "V", "M", "K", "t_out", "plan_id",
                                               "source", "trigger", "pred_first_V", "pred_V", "pred_E", "pred_omega")
            if f in s}
    st.table(pd.DataFrame({"field": list(scal), "value": [str(v) for v in scal.values()]}))

    dec = next((d for d in decs if d.get("step") == s["step"]), None)
    if dec is None:
        st.info("no decision record for this step (run the episode with keep_decisions=True)")
        return
    st.subheader("candidate ranking (decisions[*].ranking)")
    if dec.get("ranking"):
        rk = pd.DataFrame(dec["ranking"])
        show = [c for c in ("rank", "plan_id", "source", "score_U", "score_J", "U", "E", "V", "M", "K", "Omega",
                            "dev_mean") if c in rk.columns]
        _wide_table(rk[show])
    st.write({f"decisions[{k}].{f}": dec.get(f) for f in ("plan_id", "source", "trigger", "n_candidates",
                                                           "n_rejected", "committed", "ack")})
    if dec.get("macro"):
        st.subheader("committed macro-plan (decisions[*].macro)")
        st.json(dec["macro"])
    alerts = (dec.get("notes") or {}).get("alerts")
    if alerts:
        st.subheader("administrator alerts (decisions[*].notes.alerts)")
        for msg in alerts:
            st.warning(msg)
    neg = (dec.get("notes") or {}).get("negotiation")
    st.subheader("L4 dialogue (decisions[*].notes.negotiation)")
    if neg:
        st.json(neg)
    else:
        st.caption("no trade-off negotiation at this step")
    if dec.get("llm"):
        st.subheader("LLM transcripts (decisions[*].llm)")
        for t in dec["llm"]:
            st.json(t, expanded=False)
    if dec.get("rejected_sample"):
        st.subheader("rejected plans (decisions[*].rejected_sample)")
        st.json(dec["rejected_sample"], expanded=False)


main()
