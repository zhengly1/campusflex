"""Regenerate every Section 4 table from the files in ``results/`` (Table 11, statistics row).

    python -m campusflex.experiments.tables [--results DIR]

Reads whatever exists among the result artifacts and writes ``<results>/tables.md``. A
missing artifact produces "not run (artifact missing)" – no number is ever typed by hand.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..stats import group_means, paired_bootstrap, pair_rows
from .common import MAX_FALLBACK_RATE, RESULTS, compare_arms, config_conflict, config_hashes, llm_stats_by_arm

MISSING = "_not run (artifact missing)_"
POLICY_LABEL = {
    "fixed": "Fixed opening",
    "threshold": "Demand threshold",
    "threshold_eff": "Efficiency-ranked threshold",
    "planner": "Simulation planner",
    "planner_nomove": "No relocation cost",
    "A": "A masks only (v1)",
    "B": "B masks + rule-staged",
    "C": "C masks + LLM-staged",
    "D": "D masks + rule + LLM",
}
V1_ORDER = ("fixed", "threshold", "planner", "planner_nomove")


# ----------------------------------------------------------------------- utilities
def load(results: Path, rel: str) -> Any | None:
    p = results / rel
    if not p.exists():
        return None
    try:
        if p.suffix == ".csv":
            with p.open() as f:
                return list(csv.DictReader(f))
        return json.loads(p.read_text())
    except (ValueError, OSError) as e:
        return {"_error": f"unreadable: {e}"}


def rows_of(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, dict):
        r = obj.get("rows")
        return [x for x in r if isinstance(x, dict)] if isinstance(r, list) else []
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    return []


def meta_of(obj: Any) -> dict[str, Any]:
    return obj.get("meta", {}) if isinstance(obj, dict) and isinstance(obj.get("meta"), dict) else {}


def fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.{nd}f}"
    if isinstance(x, (dict, list)):
        s = json.dumps(x, default=str)
        return s if len(s) <= 80 else s[:77] + "..."
    return str(x).replace("|", "\\|").replace("\n", " ")


def md_table(header: list[str], rows: Iterable[list[Any]], nd: int = 3) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(fmt(x, nd) for x in r) + " |")
    return out


def dicts_table(items: list[dict[str, Any]], max_cols: int = 10) -> list[str]:
    keys: list[str] = []
    for it in items:
        for k in it:
            if k not in keys and not str(k).startswith("_"):
                keys.append(k)
    keys = keys[:max_cols]
    return md_table(keys, ([it.get(k) for k in keys] for it in items))


def generic_section(obj: Any) -> list[str]:
    """Render an unknown result schema: protocol meta, scalar summary, lists of records."""
    if obj is None:
        return [MISSING]
    if isinstance(obj, dict) and "_error" in obj:
        return [f"_{obj['_error']}_"]
    out: list[str] = []
    m = meta_of(obj)
    if m:
        out.append("Meta: " + ", ".join(f"{k}={fmt(v)}" for k, v in m.items()))
        out.append("")
    if isinstance(obj, dict):
        scalars = {k: v for k, v in obj.items() if not isinstance(v, (dict, list)) and k != "meta"}
        for key in ("summary", "metrics", "totals", "verdict"):
            if isinstance(obj.get(key), dict):
                scalars.update({f"{key}.{k}": v for k, v in obj[key].items() if not isinstance(v, (dict, list))})
        if scalars:
            out += md_table(["field", "value"], ([k, v] for k, v in scalars.items()))
            out.append("")
        for k, v in obj.items():
            if k in ("rows", "meta") or not isinstance(v, list) or not v or not all(isinstance(x, dict) for x in v):
                continue
            out.append(f"**{k}**")
            out.append("")
            out += dicts_table(v[:40])
            out.append("")
        n = len(rows_of(obj))
        if n:
            out.append(f"Episode/record rows in file: {n}.")
    return out or ["_empty artifact_"]


def protocol_note(obj: Any) -> list[str]:
    m = meta_of(obj)
    if m.get("protocol") == "reduced" or m.get("reduced"):
        return ["**Reduced protocol** – results are labelled as such; the full protocol is listed as not run.", ""]
    return []


# --------------------------------------------------------------------- sections
def section_v1(results: Path) -> list[str]:
    obj = load(results, "v1/summary.json")
    out = ["## Table 12 – v1 results of this implementation (means of episode-level metrics per policy)", ""]
    rows = rows_of(obj)
    if not rows:
        return out + [MISSING, ""]
    out += protocol_note(obj)
    g = group_means(rows, ("policy",), ("kwh", "service_pct", "dev_c", "moves", "switches", "score"))
    order = {p: k for k, p in enumerate(V1_ORDER)}
    g.sort(key=lambda r: order.get(r["policy"], 99))
    out += md_table(
        ["Policy", "n", "kWh", "Service %", "Dev. °C", "Moves", "Switches", "Score"],
        ([POLICY_LABEL.get(r["policy"], r["policy"]), r["n"], r["kwh"], r["service_pct"], r["dev_c"], r["moves"],
          r["switches"], None if r["policy"] == "planner_nomove" else r["score"]] for r in g),
    )
    out += ["", "Score is the full-evening Equation 4 without Ω under reference weights; – for the ablation.", ""]
    pols = {r["policy"] for r in rows}
    if "planner" in pols:
        out.append("**Paired energy saving of the planner (kWh, 95% bootstrap interval, paired by scenario and seed)**")
        out.append("")
        tab = []
        for base in ("fixed", "threshold", "threshold_eff"):
            if base not in pols:
                continue
            a, b, keys = pair_rows(rows, "policy", base, "planner", metric="kwh")
            if keys:
                bs = paired_bootstrap(a, b)
                tab.append([f"planner vs {POLICY_LABEL.get(base, base)}", bs["n"], bs["mean_diff"], bs["lo"], bs["hi"]])
        out += md_table(["Comparison", "pairs", "saving kWh", "lo", "hi"], tab)
        out.append("")
        if "planner_nomove" in pols:
            mp = np.mean([r["moves"] for r in rows if r["policy"] == "planner"])
            mn = np.mean([r["moves"] for r in rows if r["policy"] == "planner_nomove"])
            if mn > 0:
                out.append(f"Relocation cost cuts moves by {100 * (1 - mp / mn):.1f}% relative to the ablated planner "
                           f"({mp:.2f} vs {mn:.2f} moves per episode).")
                out.append("")
    out.append("**Scenario-level means (Figure 3 data)**")
    out.append("")
    gs = group_means(rows, ("scenario", "policy"), ("kwh", "dev_c", "moves", "score"))
    out += md_table(["Scenario", "Policy", "n", "kWh", "Dev. °C", "Moves", "Score"],
                    ([r["scenario"], r["policy"], r["n"], r["kwh"], r["dev_c"], r["moves"], r["score"]] for r in gs))
    return out + [""]


def _num(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def section_vgap(results: Path) -> list[str]:
    obj = load(results, "vgap_dev.csv")
    out = ["## Ledger item 1 – predicted vs realized occupied deviation V (development seeds)", ""]
    if obj is None:
        return out + [MISSING, ""]
    if isinstance(obj, dict):
        return out + [f"_{obj.get('_error')}_", ""]
    if not obj:
        return out + ["_empty file_", ""]
    cols = list(obj[0].keys())
    pred = next((c for c in ("pred_first_V", "pred_V", "predicted_V") if c in cols), None)
    real = next((c for c in ("real_first_V", "realized_V", "real_V", "V") if c in cols), None)
    if not (pred and real):
        return out + [f"_unrecognised columns: {', '.join(cols)}_", ""]
    by = [c for c in ("scenario", "mismatch", "lambda_Omega", "policy") if c in cols]
    hp, hr = ("pred_V_horizon", "real_V_horizon") if {"pred_V_horizon", "real_V_horizon"} <= set(cols) else (None, None)
    groups: dict[tuple, list[tuple[float, float, float | None, float | None]]] = defaultdict(list)
    skipped = 0
    for r in obj:
        p, v = _num(r.get(pred)), _num(r.get(real))
        if p is None or v is None:
            skipped += 1
            continue
        groups[tuple(r[c] for c in by)].append((p, v, _num(r.get(hp)) if hp else None, _num(r.get(hr)) if hr else None))
    tab = []
    for k in sorted(groups, key=lambda t: tuple(str(x) for x in t)):
        g = groups[k]
        xs = np.array([(a, b) for a, b, _, _ in g])
        hz = [(c, d) for _, _, c, d in g if c is not None and d is not None]
        tab.append(list(k) + [len(xs), xs[:, 0].sum(), xs[:, 1].sum(), (xs[:, 1] - xs[:, 0]).mean(),
                              float(np.mean([d - c for c, d in hz])) if hz else None])
    out.append("First committed interval and rollout horizon; gap = realized − predicted (°C·person-h), "
               "grouped by λΩ and model mismatch.")
    out.append("")
    out += md_table(by + ["steps", f"Σ {pred}", f"Σ {real}", "mean first-interval gap", "mean horizon gap"], tab, nd=4)
    if skipped:
        out += ["", f"{skipped} rows without numeric predicted/realized V were skipped."]
    out.append("")
    summ = load(results, "vgap_dev_summary.json")
    om = summ.get("omega_ablation_score_diff_lambda0_minus_declared") if isinstance(summ, dict) else None
    out.append("**Ω ablation – full-evening score, λΩ = 0 minus declared λΩ (paired by scenario, seed, mismatch)**")
    out.append("")
    if isinstance(om, dict) and om:
        out += md_table(["pairs", "mean diff", "lo", "hi", "excludes 0"],
                        [[om.get("n"), om.get("mean_diff"), om.get("lo"), om.get("hi"), om.get("excludes_zero")]])
    else:
        out.append(MISSING if summ is None else "_no Ω ablation in vgap_dev_summary.json_")
    return out + [""]


def _e1_rows(results: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ab, cd = load(results, "e1_AB.json"), load(results, "e1_CD.json")
    metas = [m for m in (meta_of(ab), meta_of(cd)) if m]
    rows = []
    for obj in (ab, cd):
        h = meta_of(obj).get("config_hash")
        for r in rows_of(obj):
            r = dict(r)
            if h and not r.get("config_hash"):
                r["config_hash"] = h
            rows.append(r)
    return rows, metas


def _arm_label(r: dict[str, Any]) -> str:
    return str(r.get("policy") or str(r.get("arm", "")).replace("arm_", ""))


def section_e1(results: Path) -> list[str]:
    out = ["## E1 – staged plans (full-evening combined score, held-out seeds)", ""]
    rows, metas = _e1_rows(results)
    cd = load(results, "e1_CD.json")
    if isinstance(cd, dict) and cd.get("status") in ("not_run", "aborted"):
        out += [f"Arms C and D: **{cd['status'].replace('_', ' ')}** – {fmt(cd.get('reason'))}.", ""]
    if not rows:
        return out + [MISSING, ""]
    for m in metas:
        if m.get("protocol") == "reduced" or m.get("reduced"):
            out += ["**Reduced E1 protocol** – fault and fault-during-surge only, no model mismatch, one repeat.", ""]
            break
    for m in metas:
        if m.get("frozen") is False:
            out += [f"**Not frozen** – {m.get('experiment', 'E1')} ran without results/frozen_config.json "
                    f"(config {m.get('config_hash')}).", ""]
    for r in rows:
        r.setdefault("mismatch", False)
        r["policy"] = _arm_label(r)
    arms = sorted({r["policy"] for r in rows})
    llm_stats = llm_stats_by_arm([r for r in rows if r["policy"] in ("C", "D")], arm_key="policy")
    # An LLM arm that mostly fell back is not an LLM result: its means are not shown either.
    invalid = {a for a, st in llm_stats.items() if st["status"] != "run"}
    shown = [r for r in rows if r["policy"] not in invalid]
    g = group_means(shown, ("scenario", "mismatch", "policy"), ("score", "dev_c", "moves", "kwh", "unserved_person_min"))
    out += md_table(["Scenario", "Mismatch", "Arm", "n", "Score", "Dev. °C", "Moves", "kWh", "Unserved p-min"],
                    ([r["scenario"], r["mismatch"], r["policy"], r["n"], r["score"], r["dev_c"], r["moves"], r["kwh"],
                      r["unserved_person_min"]] for r in g))
    out.append("")
    for a in sorted(invalid):
        out += [f"Arm {a} rows omitted from the means: {llm_stats[a]['status']} (not an LLM result).", ""]
    if llm_stats:
        out.append(f"**LLM plan statistics** (an arm with fallback or failed-call rate > {MAX_FALLBACK_RATE} is not an "
                   "LLM result and is not compared)")
        out.append("")
        out += md_table(["Arm", "episodes", "L2 calls", "failed calls", "fallback rate", "valid proposal rate",
                         "LLM plan selection rate", "status"],
                        ([a, s["episodes"], s["n_calls"], s["n_failed_calls"], s["fallback_rate"], s["valid_proposal_rate"],
                          s["llm_selection_rate"], s["status"]] for a, s in llm_stats.items()), nd=4)
        out.append("")
    hashes = config_hashes(rows, "policy")
    comps = [(x, "B") for x in ("C", "D") if x in arms] + ([("B", "A")] if {"A", "B"} <= set(arms) else [])
    if comps:
        out.append("**Paired comparisons (arm − reference; negative = better; LLM repeats averaged per pair; "
                   "Eq. 4 order: unserved person-minutes first, then score)**")
        out.append("")
        tab, skipped = [], []
        for x, y in comps:
            bad = [a for a in (x, y) if a in llm_stats and llm_stats[a]["status"] != "run"]
            if bad:
                why = ", ".join(a + " " + llm_stats[a]["status"] for a in bad)
                skipped.append(f"{x} − {y}: not computed ({why})")
                continue
            conflict = config_conflict(hashes, x, y)
            if conflict:
                skipped.append(f"{x} − {y}: **refused** – {conflict}")
                continue
            for sc in sorted({r["scenario"] for r in rows}) + ["(all)"]:
                sub = rows if sc == "(all)" else [r for r in rows if r["scenario"] == sc]
                c = compare_arms(sub, "policy", x, y)
                if c:
                    tab.append([sc, f"{x} − {y}", c["n"], c["mean_diff"], c["lo"], c["hi"], c["excludes_zero"],
                                c["unserved"]["mean_diff"], c["decided_by"], c["a_better"]])
        if tab:
            out += md_table(["Scenario", "Comparison", "pairs", "score diff", "lo", "hi", "excludes 0",
                             "unserved diff", "decided by", "arm better"], tab)
            out.append("")
        for s in skipped:
            out.append(f"- {s}")
        if skipped:
            out.append("")
    return out


def _llm_stats_row(name: str, arm: dict[str, Any]) -> list[Any]:
    s = arm.get("llm") or {}
    return [name, arm.get("status"), arm.get("reason") or arm.get("model") or arm.get("note"), s.get("n_calls"),
            s.get("n_failed_calls"), s.get("fallback_rate"), s.get("valid_proposal_rate")]


LLM_STATS_HEADER = ["Arm", "status", "reason / model", "LLM calls", "failed calls", "fallback rate", "valid rate"]


def _bad(obj: Any) -> list[str] | None:
    if obj is None:
        return [MISSING, ""]
    if isinstance(obj, dict) and "_error" in obj:
        return [f"_{obj['_error']}_", ""]
    if not isinstance(obj, dict):
        return ["_unrecognised artifact_", ""]
    return None


def _meta_lines(obj: dict[str, Any]) -> list[str]:
    m = meta_of(obj)
    out = protocol_note(obj)
    if m.get("frozen") is False:
        out += [f"**Not frozen** – run without results/frozen_config.json (config {m.get('config_hash')}).", ""]
    return out


def section_e2(results: Path) -> list[str]:
    obj = load(results, "e2.json")
    out = ["## E2 – intent compilation (L1 vs structured form)", ""]
    bad = _bad(obj)
    if bad:
        return out + bad
    out += _meta_lines(obj)
    arms = obj.get("arms") or {}
    out.append(f"Requests: {fmt(obj.get('n_requests'))}.")
    out.append("")
    out += md_table(LLM_STATS_HEADER, (_llm_stats_row(k, v) for k, v in arms.items()), nd=4)
    out.append("")
    summ = {k: v["summary"] for k, v in arms.items() if isinstance(v.get("summary"), dict)}
    if summ:
        fields = ("n", "valid_share", "field_accuracy", "hard_constraint_omissions", "weight_levels_correct_share",
                  "weight_direction_correct_share_nondefault", "clarification_correct_share", "max_questions")
        out += md_table(["Arm"] + list(fields) + ["zero omissions", "direction ≥ 90%"],
                        ([k] + [s.get(f) for f in fields] + [s.get("success", {}).get("zero_hard_omissions"),
                                                             s.get("success", {}).get("weight_direction_ge_90pct")]
                         for k, s in summ.items()))
        out.append("")
        cats = sorted({c for s in summ.values() for c in (s.get("field_accuracy_by_category") or {})})
        if cats:
            out.append("**Field accuracy by request category**")
            out.append("")
            out += md_table(["Arm"] + cats, ([k] + [(s.get("field_accuracy_by_category") or {}).get(c) for c in cats]
                                             for k, s in summ.items()))
            out.append("")
    pdir = obj.get("plan_direction") or {}
    if pdir:
        out.append("**Plan direction when compiled weight levels are applied**")
        out.append("")
        out += md_table(["Arm", "scenario", "seeds", "checks", "intended share"],
                        ([k, v.get("scenario"), v.get("seeds"), len(v.get("checks") or []), v.get("intended_share")]
                         for k, v in pdir.items()))
        out.append("")
    return out


E3_METRICS = ("unserved_person_min", "moves", "forecast_mae_30min", "kwh", "dev_c", "score")


def section_e3(results: Path) -> list[str]:
    obj = load(results, "e3.json")
    out = ["## E3 – notice interpretation (constant / keyword / LLM / oracle)", ""]
    bad = _bad(obj)
    if bad:
        return out + bad
    out += _meta_lines(obj)
    m = meta_of(obj)
    out.append(f"Seeds: {fmt(obj.get('seeds') or m.get('seeds'))}; scenarios: {fmt(obj.get('scenarios') or m.get('scenarios'))}; "
               f"planner: {fmt(obj.get('planner'))}; LLM repeats: {fmt(m.get('llm_repeats'))}, "
               f"caching: {fmt(m.get('llm_caching'))}.")
    out.append("")
    status = obj.get("status") or {}
    reasons = obj.get("status_reasons") or {}
    ls = obj.get("llm_stats") or {}
    tab = []
    for a, st in (status.items() if isinstance(status, dict) else []):
        s_ = ls if a == "llm" else {}
        tab.append([a, st, reasons.get(a), s_.get("n_calls"), s_.get("n_failed_calls"), s_.get("fallback_rate"),
                    s_.get("valid_proposal_rate")])
    out += md_table(LLM_STATS_HEADER, tab, nd=4)
    out.append("")
    ov = obj.get("overall") or {}
    if ov:
        out.append("**Overall means per arm**")
        out.append("")
        out += md_table(["Arm", "n"] + list(E3_METRICS), ([a, v.get("n")] + [v.get(k) for k in E3_METRICS]
                                                         for a, v in ov.items()))
        out.append("")
    ps = obj.get("per_scenario") or []
    if ps:
        out.append("**Per scenario**")
        out.append("")
        out += md_table(["Scenario", "Arm", "n"] + list(E3_METRICS),
                        ([r.get("scenario"), r.get("arm"), r.get("n")] + [r.get(k) for k in E3_METRICS] for r in ps))
        out.append("")
    pr = obj.get("paired") or {}
    if pr:
        out.append("**Paired bootstrap vs constant forecast (arm − constant; repeats averaged per scenario/seed)**")
        out.append("")
        out += md_table(["Comparison", "pairs", "mean diff", "lo", "hi", "excludes 0"],
                        ([k] + ([v.get("n"), v.get("mean_diff"), v.get("lo"), v.get("hi"), v.get("excludes_zero")]
                                if v else [None, None, None, None, "not evaluable"]) for k, v in pr.items()))
        out.append("")
    gc = obj.get("gap_closed") or {}
    out.append("**Gap closed by the LLM: (keyword − LLM) / (keyword − oracle); success = at least half on every metric**")
    out.append("")
    if "status" in gc and len(gc) <= 2:
        out.append(f"_not computed: LLM arm {gc.get('status')}" + (f" – {gc['reason']}" if gc.get("reason") else "") + "_")
    else:
        out += md_table(["Metric", "keyword", "LLM", "oracle", "closed", "note"],
                        ([k, v.get("keyword"), v.get("llm"), v.get("oracle"), v.get("closed"), v.get("note")]
                         for k, v in gc.items() if isinstance(v, dict)))
        out += ["", f"Success (≥ 0.5 on every metric with a gap): {fmt(gc.get('success_at_least_half'))}"]
    out.append("")
    n = len(rows_of(obj))
    if n:
        out += [f"Episode rows in file: {n}.", ""]
    return out


def section_e4(results: Path) -> list[str]:
    obj = load(results, "e4.json")
    out = ["## E4 – explanation faithfulness", ""]
    bad = _bad(obj)
    if bad:
        return out + bad
    out += _meta_lines(obj)
    out.append(f"Decisions from: {fmt(obj.get('source'))}; episodes: {fmt(obj.get('n_episodes'))}.")
    out.append("")
    arms = obj.get("arms") or {}
    out += md_table(LLM_STATS_HEADER, (_llm_stats_row(k, v) for k, v in arms.items()), nd=4)
    out.append("")
    summ = {k: v["summary"] for k, v in arms.items() if isinstance(v.get("summary"), dict)}
    if summ:
        fields = ("n_texts", "n_numbers", "n_matched", "n_flagged", "share_matched", "texts_fully_matched_share",
                  "displayed_unverified_numbers", "success_100pct_matched_or_flagged")
        out += md_table(["Arm"] + list(fields), ([k] + [s.get(f) for f in fields] for k, s in summ.items()))
        out.append("")
    return out


def section_generic(results: Path, rel: str, title: str) -> list[str]:
    obj = load(results, rel)
    return [f"## {title}", ""] + protocol_note(obj) + generic_section(obj) + [""]


def section_e6(results: Path) -> list[str]:
    obj = load(results, "e6.json")
    out = ["## E6 – L5 review vs random search (seeds 20–29)", ""]
    if obj is None:
        return out + [MISSING, ""]
    out += _meta_lines(obj) if isinstance(obj, dict) else []
    arms = obj.get("arms", {})
    out += md_table(["Arm", "status", "source / reason", "config", "LLM calls", "failed calls", "fallback rate"],
                    ([k, v.get("status"), v.get("source") or v.get("reason"), v.get("config_hash"),
                      (v.get("llm") or {}).get("n_calls"), (v.get("llm") or {}).get("n_failed_calls"),
                      (v.get("llm") or {}).get("fallback_rate")] for k, v in arms.items()))
    out.append("")
    pm = obj.get("per_arm_means") or {}
    if pm:
        out += md_table(["Arm", "Unserved p-min", "Score", "kWh", "Dev. °C", "Moves"],
                        ([k, v.get("unserved_person_min"), v.get("score"), v.get("kwh"), v.get("dev_c"), v.get("moves")]
                         for k, v in pm.items()))
        out.append("")
    comps = obj.get("comparisons") or {}
    tab = [[k, v.get("n"), (v.get("unserved") or {}).get("mean_diff"), (v.get("unserved") or {}).get("lo"),
            (v.get("unserved") or {}).get("hi"), v.get("mean_diff"), v.get("lo"), v.get("hi"), v.get("excludes_zero"),
            v.get("decided_by")]
           for k, v in comps.items() if v]
    tab += [[k, None, None, None, None, None, None, None, "not evaluable", None] for k, v in comps.items() if not v]
    if tab:
        out.append("Paired comparisons in Eq. 4 order: unserved person-minutes (U) first, then score.")
        out.append("")
        out += md_table(["Comparison", "pairs", "U diff", "U lo", "U hi", "score diff", "lo", "hi", "excludes 0",
                         "decided by"], tab)
        out.append("")
    v = obj.get("verdict") or {}
    if v:
        out += md_table(["verdict field", "value"], ([k, x] for k, x in v.items()))
        out.append("")
    rv = obj.get("reviews") or {}
    if rv:
        out += md_table(["Review", "proposals", "accepted", "review acceptance rate", "protocol"],
                        ([k, x.get("proposals"), x.get("accepted"), x.get("review_acceptance_rate"), x.get("protocol")]
                         for k, x in rv.items()))
        out.append("")
    return out


def section_t1(results: Path) -> list[str]:
    obj = load(results, "trace_T1.json")
    out = ["## T1 – integrated case trace", ""]
    if obj is None:
        return out + [MISSING, ""]
    rec = obj.get("record") if isinstance(obj, dict) and isinstance(obj.get("record"), dict) else obj
    if isinstance(rec, dict):
        flat = []
        for k in ("step", "state_version", "committed", "counterfactuals", "jointly_decisive", "verdict"):
            if k in rec:
                flat.append([k, rec[k]])
        if isinstance(rec.get("inputs"), dict):
            for k, v in rec["inputs"].items():
                flat.append([f"inputs.{k}", v])
        if flat:
            out += md_table(["field", "value"], flat)
            out.append("")
            return out
    return out + generic_section(obj) + [""]


def section_smart(results: Path) -> list[str]:
    """Smart device extension (not part of Table 16; simulated occupants and sensors)."""
    obj = load(results, "smart/summary.json")
    out = ["## Smart device extension – per-device IoT dispatch, learned profiles, sensor control", ""]
    if not isinstance(obj, dict) or "eval" not in obj:
        return out + [MISSING, ""]
    m = obj.get("meta", {})
    out += [f"{m.get('train_days')} training and {m.get('eval_days')} evaluation evenings (two injected sensor "
            "faults each); space plan from the core planner (arm A). Discomfort = share of occupied person-minutes.", ""]
    out += md_table(
        ["Arm", "kWh", "thermal %", "light %", "CO2>1000 %", "glare %", "manual adj./evening"],
        ([a, s["kwh"], s["discomfort_pct"]["thermal"], s["discomfort_pct"]["light"], s["discomfort_pct"]["co2"],
          s["discomfort_pct"]["glare"], s["adjustments_per_evening"]] for a, s in obj["eval"].items()), nd=2)
    out.append("")
    rows = []
    for name, c in obj.get("comparisons", {}).items():
        for metric, v in c.items():
            rows.append([name, metric, v.get("mean_diff"), v.get("lo"), v.get("hi")])
    out += ["**Paired by evaluation evening (smart − other, 95 % bootstrap interval)**", ""]
    out += md_table(["Comparison", "metric", "mean diff", "lo", "hi"], rows, nd=3)
    pa, ad, ti = obj.get("profile_accuracy", {}), obj.get("anomaly_detection", {}), obj.get("thermal_identification", {})
    out += ["", f"Profiles vs latent habits ({pa.get('cells_with_room_evidence')}/{pa.get('cells_total')} room-slots "
            f"with own evidence): setpoint MAE {pa.get('setpoint_mae_c')} °C, lux MAE {pa.get('lux_mae')}, "
            f"curtain MAE {pa.get('curtain_mae_pct')} %, fan mismatch {pa.get('fan_mismatch_share')}, "
            f"projector mismatch {pa.get('projector_mismatch_share')}.",
            f"Sensor faults (stuck/drift): {ad.get('detected')}/{ad.get('faults')} detected, {ad.get('false_alarms')} "
            f"false alarms, mean delay {ad.get('mean_delay_min')} min. Identified thermal model vs Table 6: median "
            f"relative error R {ti.get('R', {}).get('median_rel_error')}, C {ti.get('C', {}).get('median_rel_error')}.",
            ""]
    return out


def build_tables(results: Path = RESULTS) -> str:
    results = Path(results)
    parts: list[str] = ["# CampusFlex – Section 4 tables (regenerated from results/)", "",
                        "All values are simulated estimates from the synthetic six-room campus.", ""]
    parts += section_v1(results)
    parts += section_vgap(results)
    parts += section_e1(results)
    parts += section_e2(results)
    parts += section_e3(results)
    parts += section_e4(results)
    parts += section_generic(results, "e5_scripted.json", "E5 – scripted adversarial suite (no live LLM)")
    parts += section_generic(results, "e5_live.json", "E5 – live injected notices")
    parts += section_e6(results)
    parts += section_t1(results)
    parts += section_smart(results)
    return "\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description="Regenerate Section 4 tables from result files")
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = Path(a.results)
    text = build_tables(res)
    out = Path(a.out) if a.out else res / "tables.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
