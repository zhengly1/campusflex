"""Evidence ledger (Section 4.8, Table 16). Status is computed from the files on disk at the
time the script runs – never from memory.

    python -m campusflex.experiments.evidence_ledger [--root DIR]

Statuses: Executed · Partial (reduced protocol, only some artifacts/arms, an arm not run or
invalid because its LLM calls mostly fell back, a held-out run without the frozen
configuration, a protocol shortfall in seeds/episodes, or a skipped real-broker test) · Not run ·
Not started (process records still templates).
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..config import HELDOUT_SEEDS
from .common import MAX_FALLBACK_RATE, ROOT, llm_stats_by_arm

TEMPLATE_MARKER = "<!-- TEMPLATE"


@dataclass
class Item:
    num: int
    name: str
    llm: bool
    artifacts: list[str]
    check: Callable[[Path, "Item"], tuple[str, str]] | None = None
    notes: list[str] = field(default_factory=list)


def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except (ValueError, OSError):
        return None


def _is_reduced(obj: Any) -> bool:
    m = obj.get("meta", {}) if isinstance(obj, dict) else {}
    return isinstance(m, dict) and (m.get("protocol") == "reduced" or bool(m.get("reduced")))


def _json_status(root: Path, item: Item) -> tuple[str, str]:
    present = [a for a in item.artifacts if (root / a).exists()]
    if not present:
        return "Not run", "no artifact"
    notes = []
    status = "Executed" if len(present) == len(item.artifacts) else "Partial"
    if status == "Partial":
        notes.append("missing: " + ", ".join(a for a in item.artifacts if a not in present))
    for a in present:
        if a.endswith(".json"):
            obj = _read_json(root / a)
            if obj is None:
                return "Partial", f"{a} unreadable"
            if isinstance(obj, dict) and obj.get("status") in ("not_run", "aborted"):
                status = "Partial" if status == "Executed" and len(present) > 1 else "Not run"
                why = str(obj.get("reason") or "")
                why = why.split(" (request_id")[0].split("{'error': {'message': ")[-1].strip("'\" ")
                notes.append(f"{a}: status {obj['status']}" + (f" ({why[:110]})" if why else ""))
            if _is_reduced(obj):
                status = "Partial"
                notes.append(f"{a}: reduced protocol")
            st = obj.get("status") if isinstance(obj, dict) else None
            if isinstance(st, dict):                       # e.g. E3: {"constant": "run", "llm": "not_run"}
                nr = [f"{k} ({v})" for k, v in st.items() if v != "run"]
                if nr:
                    status = "Partial"
                    notes.append(f"{a}: arm(s) not run or invalid: {', '.join(nr)}")
            arms = obj.get("arms") if isinstance(obj, dict) else None
            if isinstance(arms, dict):
                nr = [f"{k} ({v.get('status')})" for k, v in arms.items()
                      if isinstance(v, dict) and v.get("status") != "run"]
                if nr:
                    status = "Partial"
                    notes.append(f"{a}: arm(s) not run or invalid: {', '.join(nr)}")
            m = obj.get("meta") if isinstance(obj, dict) else None
            if isinstance(m, dict):
                if m.get("frozen") is False:
                    status = "Partial"
                    notes.append(f"{a}: not run under a frozen configuration (freeze gate)")
                for k, what in (("prompts_frozen", "prompts_note"), ("notices_frozen", "notices_note")):
                    if m.get(k) is False:
                        status = "Partial"
                        notes.append(f"{a}: {m.get(what) or k + ' is false'} (freeze gate)")
                if m.get("failed_jobs"):
                    status = "Partial"
                    notes.append(f"{a}: {len(m['failed_jobs'])} failed episode(s)")
    return status, "; ".join(notes)


def _add(status: str, notes: str, extra: list[str]) -> tuple[str, str]:
    if not extra:
        return status, notes
    if status == "Executed":
        status = "Partial"
    return status, "; ".join([n for n in [notes] + extra if n])


def _e1_protocol_issues(obj: Any, rel: str, arms: str) -> list[str]:
    """Held-out seeds 10–19, frozen configuration and the episode count of e1._grid."""
    from .e1 import expected_episodes

    if not isinstance(obj, dict) or obj.get("status") in ("not_run", "aborted"):
        return []
    m = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
    rows = obj.get("rows") if isinstance(obj.get("rows"), list) else []
    out = []
    if m.get("seeds") != list(HELDOUT_SEEDS):
        out.append(f"{rel}: seeds {m.get('seeds')} are not the held-out seeds 10–19")
    if m.get("frozen") is not True and "frozen" not in m:
        out.append(f"{rel}: no frozen-configuration record")
    exp = expected_episodes(arms, bool(m.get("reduced") or m.get("protocol") == "reduced"))
    if len(rows) != exp:
        out.append(f"{rel}: {len(rows)} episode rows, protocol needs {exp}")
    if arms == "CD":
        for arm, st in llm_stats_by_arm([r for r in rows if isinstance(r, dict)]).items():
            if st["status"] != "run":
                fb = st.get("fallback_rate")
                out.append(f"{rel}: {arm} {st['status']}"
                           + (f" (fallback rate {fb:.2f} > {MAX_FALLBACK_RATE})" if fb is not None else ""))
    return out


def _ran(obj: Any) -> bool:
    return isinstance(obj, dict) and obj.get("status") not in ("not_run", "aborted")


def _e1_hashes(obj: Any) -> set[str]:
    if not isinstance(obj, dict):
        return set()
    m = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
    hs = {r.get("config_hash") for r in obj.get("rows") or [] if isinstance(r, dict)}
    return {h for h in hs | {m.get("config_hash")} if h}


def _e5_live_issues(obj: Any, rel: str) -> list[str]:
    """Each variant must evaluate enough runs to claim its target (``share_evaluated``)."""
    if not _ran(obj):
        return []
    out = []
    for v, s in ((obj.get("summary") or {}).get("variants") or {}).items():
        if isinstance(s, dict) and s.get("target_met") is None:
            out.append(f"{rel}: variant {v} {s.get('target_status') or 'target not claimed'}")
    return out


def _e1_status(arms: str) -> Callable[[Path, "Item"], tuple[str, str]]:
    def check(root: Path, item: "Item") -> tuple[str, str]:
        s, n = _json_status(root, item)
        rel = item.artifacts[0]
        p = root / rel
        extra: list[str] = []
        if arms == "CD":
            live = item.artifacts[1]
            extra += _e5_live_issues(_read_json(root / live), live) if (root / live).exists() else []
        if s == "Not run" or not p.exists():
            return _add(s, n, extra) if s != "Not run" else (s, n)
        obj = _read_json(p)
        extra = _e1_protocol_issues(obj, rel, arms) + extra
        if arms == "CD" and _ran(obj):
            # C/D are compared with B: both files must come from one configuration.
            ab = _read_json(root / "results" / "e1_AB.json")
            hs = _e1_hashes(obj) | _e1_hashes(ab)
            if len(hs) > 1:
                extra.append(f"{rel}: config_hash differs from results/e1_AB.json ({', '.join(sorted(hs))}); "
                             "C/D cannot be compared with B")
        return _add(s, n, extra)
    return check


def _expect(m: dict[str, Any], key: str, want: Any, rel: str, what: str) -> list[str]:
    got = m.get(key)
    return [] if got == want else [f"{rel}: {key} {got} – the protocol needs {what}"]


def _protocol_items_6_8(root: Path, item: "Item") -> tuple[str, str]:
    """Seeds, scenarios, repeats and fresh calls of T1, E2, E3, E4 (Table 13)."""
    from ..config import HELDOUT_SEEDS as HS
    from ..scenarios import E1_SCENARIOS, NOTICE_SCENARIOS
    from .e3 import LLM_REPEATS

    s, n = _json_status(root, item)
    extra: list[str] = []
    for rel in item.artifacts:
        obj = _read_json(root / rel) if (root / rel).exists() else None
        if not _ran(obj):
            continue
        m = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
        name = Path(rel).name
        if name == "trace_T1.json" and m.get("seed") not in HS:
            extra.append(f"{rel}: seed {m.get('seed')} is not a held-out seed")
        elif name == "e2.json":
            n_req, llm = obj.get("n_requests"), (obj.get("arms") or {}).get("llm_compiler") or {}
            if n_req != 40:
                extra.append(f"{rel}: {n_req} requests, the labelled set has 40")
            if llm.get("status") == "run" and len(llm.get("rows") or []) != n_req:
                extra.append(f"{rel}: LLM compiler scored {len(llm.get('rows') or [])} of {n_req} requests")
        elif name == "e3.json":
            extra += _expect(m, "seeds", list(HS), rel, "held-out seeds 10–19")
            extra += _expect(m, "scenarios", list(NOTICE_SCENARIOS), rel, f"all notice scenarios {list(NOTICE_SCENARIOS)}")
            if (obj.get("status") or {}).get("llm") == "run":
                extra += _expect(m, "llm_repeats", LLM_REPEATS, rel, f"{LLM_REPEATS} uncached repeats")
                extra += _expect(m, "llm_caching", "disabled", rel, "uncached LLM calls")
        elif name == "e4.json":
            extra += _expect(m, "seeds", list(HS[:2]), rel, "fresh decisions on held-out seeds 10–11")
            extra += _expect(m, "scenarios", list(E1_SCENARIOS + NOTICE_SCENARIOS), rel, "every E1 and notice scenario")
    return _add(s, n, extra)


def _freeze_drift(root: Path, rel: str, obj: Any) -> list[str]:
    """An artifact of a held-out item must match the *current* freeze record: a later
    ``freeze --force`` (other config, prompts, model or notices) makes it stale."""
    from ..llm.prompts import changed_roles

    rec = _read_json(root / "results" / "frozen_config.json")
    m = obj.get("meta") if isinstance(obj, dict) else None
    if not isinstance(rec, dict) or not isinstance(m, dict) or not _ran(obj):
        return []
    out = []
    if m.get("config_hash") and rec.get("config_hash") and m["config_hash"] != rec["config_hash"]:
        out.append(f"{rel}: config_hash {m['config_hash']} differs from the freeze record {rec['config_hash']}")
    llm = rec.get("llm") or {}
    if isinstance(m.get("prompt_hashes"), dict) and llm.get("prompt_hashes"):
        ch = changed_roles(m["prompt_hashes"], llm["prompt_hashes"])
        if ch:
            out.append(f"{rel}: prompts {', '.join(ch)} differ from the freeze record")
    mod = m.get("model")
    if mod and llm.get("model") and mod != llm["model"]:
        out.append(f"{rel}: model {mod} differs from the frozen model {llm['model']}")
    if m.get("notices_hash") and rec.get("notices_hash") and m["notices_hash"] != rec["notices_hash"]:
        out.append(f"{rel}: held-out notices differ from the freeze record")
    return out


def _v1_status(root: Path, item: Item) -> tuple[str, str]:
    s, n = _json_status(root, item)
    tr = root / "results" / "test_results.txt"
    if tr.exists():
        txt = tr.read_text(errors="replace")
        m = re.search(r"(\d+) passed", txt)
        failed = re.search(r"(\d+) failed", txt)
        extra = f"tests: {m.group(1) if m else '?'} passed" + (f", {failed.group(1)} failed" if failed else "")
        n = f"{n}; {extra}" if n else extra
        if failed and s == "Executed":
            s = "Partial"
    return s, n


def _iot_status(root: Path, item: Item) -> tuple[str, str]:
    test = root / "tests" / "test_iot.py"
    if not test.exists():
        return "Not started", "tests/test_iot.py missing"
    tr = root / "results" / "test_results.txt"
    if tr.exists():
        txt = tr.read_text(errors="replace")
        lines = [l for l in txt.splitlines() if "test_iot" in l]
        if lines and any(("FAILED" in l or "ERROR" in l) for l in lines):
            return "Partial", "test_iot.py has failures in results/test_results.txt"
        on_broker = [l for l in lines if ("mqtt" in l.lower() or "broker" in l.lower()) and "unreachable" not in l.lower()]
        broker = [l for l in on_broker if "SKIPPED" in l]
        if broker:
            return "Partial", ("test_iot.py passed, but the real-MQTT broker test was SKIPPED – only the "
                               "in-process bus was exercised")
        if lines:
            real = [l for l in on_broker if "PASSED" in l]
            return "Executed", ("test_iot.py passed in results/test_results.txt"
                                + (f", incl. {len(real)} test(s) through a real MQTT broker" if real else ""))
    return "Not run", "test file exists; no recorded run in results/test_results.txt"


def _docs_status(root: Path, item: Item) -> tuple[str, str]:
    reviews = root / "docs" / "reviews.md"
    contrib = root / "docs" / "contrib.md"
    disag = root / "docs" / "disagreements.md"
    if not reviews.exists() and not contrib.exists():
        return "Not started", "no files"
    n_rev = 0
    reviewers: dict[str, int] = {}
    if reviews.exists():
        for line in reviews.read_text().splitlines():
            m = re.match(r"\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*(M\d)\s*(?:→|->)\s*(M\d)\s*\|", line)
            if m:
                n_rev += 1
                reviewers[m.group(2)] = reviewers.get(m.group(2), 0) + 1
    n_dis = 0
    if disag.exists():
        n_dis = len(re.findall(r"^\*\*Decision date:\*\*\s*\d{4}-\d{2}-\d{2}", disag.read_text(), flags=re.M))
    signed = 0
    if contrib.exists():
        signed = len(re.findall(r"^\|\s*M\d\s*\|.*\|\s*signed\s*\|\s*\d{4}-\d{2}-\d{2}\s*\|", contrib.read_text(), flags=re.M | re.I))
    note = f"{n_rev} dated review records; {len(reviewers)} distinct reviewers; {n_dis} disagreement records; {signed}/5 sign-offs"
    if n_rev == 0 and n_dis == 0 and signed == 0:
        return "Not started", note
    ok = n_rev >= 10 and len(reviewers) >= 5 and all(v >= 2 for v in reviewers.values()) and n_dis >= 1 and signed == 5
    return ("Executed" if ok else "Partial"), note


ITEMS: list[Item] = [
    Item(0, "v1 core, 160 episodes, 13 tests", False, ["results/v1/summary.json"], _v1_status),
    Item(1, "Predicted-vs-realized V log; Ω term", False, ["results/vgap_dev.csv"]),
    Item(2, "MQTT device layer and its tests", False, ["tests/test_iot.py"], _iot_status),
    Item(3, "E5 scripted suite (120 fixtures)", False, ["results/e5_scripted.json"]),
    Item(4, "E1 arms A and B on held-out seeds", False, ["results/e1_AB.json"], _e1_status("AB")),
    Item(5, "E1 arms C and D; E5 live sub-test", True, ["results/e1_CD.json", "results/e5_live.json"], _e1_status("CD")),
    Item(6, "Joint trace T1 with counterfactuals", True, ["results/trace_T1.json"], _protocol_items_6_8),
    Item(7, "E2, E3, E4", True, ["results/e2.json", "results/e3.json", "results/e4.json"], _protocol_items_6_8),
    Item(8, "E6 and L5 review report (stretch)", True, ["results/e6.json", "docs/l5_report.md"]),
    Item(9, "Review records, disagreement record, contribution sign-off", False,
         ["docs/reviews.md", "docs/contrib.md"], _docs_status),
]


def ledger(root: Path = ROOT) -> list[dict[str, Any]]:
    out = []
    for it in ITEMS:
        status, note = (it.check or _json_status)(Path(root), it)
        if 4 <= it.num <= 8:                         # held-out items: checked against the freeze record
            drift = [x for a in it.artifacts if a.endswith(".json") and (Path(root) / a).exists()
                     for x in _freeze_drift(Path(root), a, _read_json(Path(root) / a))]
            status, note = _add(status, note, drift) if status != "Not run" else (status, note)
        out.append({"#": it.num, "item": it.name, "llm": "Yes" if it.llm else "No",
                    "artifacts": ", ".join(it.artifacts), "status": status, "evidence": note})
    return out


def render(rows: list[dict[str, Any]]) -> str:
    lines = ["# Evidence ledger (Table 16) – statuses computed from files on disk", "",
             "| # | Item | LLM | Artifact | Status | Evidence |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['#']} | {r['item']} | {r['llm']} | {r['artifacts']} | {r['status']} | {r['evidence']} |")
    lines += ["", "The report cites a result only when its artifact exists; items 1–4 need no API access."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description="Evidence ledger from artifacts on disk")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root)
    text = render(ledger(root))
    out = Path(a.out) if a.out else root / "results" / "evidence_ledger.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    return out


if __name__ == "__main__":
    main()
