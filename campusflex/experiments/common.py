"""Shared experiment plumbing: parallel episode jobs, result files under ``results/``, the freeze
gate, LLM-arm health statistics and Eq. 4-ordered (lexicographic) paired comparisons."""
from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..config import PlannerConfig
from ..episode import EpisodeResult, run_episode
from ..policies import make_policy
from ..scenarios import make_scenario
from ..stats import pair_rows, paired_bootstrap

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def write_json(path: str | Path, obj: Any) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, default=str))
    return p


def read_json(path: str | Path) -> Any:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return json.loads(p.read_text())


def episode_job(spec: dict[str, Any]) -> dict[str, Any]:
    """Run one non-LLM episode from a picklable spec."""
    cfg = PlannerConfig.from_dict(spec["cfg"]) if isinstance(spec.get("cfg"), dict) else (spec.get("cfg") or PlannerConfig())
    sc = make_scenario(spec["scenario"], spec["seed"], mismatch=spec.get("mismatch", False), inject=spec.get("inject", False))
    pol = make_policy(spec["policy"], cfg)
    r: EpisodeResult = run_episode(sc, pol, cfg, keep_decisions=spec.get("keep_decisions", False))
    out = r.summary()
    out.update({k: spec[k] for k in spec.get("tag_keys", [])})
    if spec.get("keep_steps"):
        out["_steps"] = r.steps
    if spec.get("keep_decisions"):
        out["_decisions"] = r.decisions
    return out


def run_parallel(fn: Callable[[dict[str, Any]], Any], specs: list[dict[str, Any]], workers: int | None = None) -> list[Any]:
    workers = workers or min(len(specs), max(1, (os.cpu_count() or 2) - 1), 32)
    if workers <= 1 or len(specs) <= 1:
        return [fn(s) for s in specs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, specs, chunksize=1))


# ------------------------------------------------------------------ freeze gate (Section 4.2)
FROZEN = RESULTS / "frozen_config.json"


def config_hash(cfg: PlannerConfig) -> str:
    return hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True).encode()).hexdigest()[:12]


def frozen_config(path: str | Path | None = None) -> tuple[PlannerConfig, str, bool]:
    """Configuration locked at the freeze gate: ``(cfg, config_hash, frozen)``.

    Without ``results/frozen_config.json`` the declared defaults are used and ``frozen`` is
    False; held-out runs still execute but are labelled ``meta.frozen = false`` and the
    evidence ledger shows them as Partial.
    """
    p = Path(path) if path is not None else FROZEN
    if not p.exists():
        cfg = PlannerConfig()
        return cfg, config_hash(cfg), False
    d = json.loads(p.read_text())
    cfg = PlannerConfig.from_dict(d["config"] if isinstance(d, dict) and isinstance(d.get("config"), dict) else d)
    h = config_hash(cfg)
    stored = d.get("config_hash") if isinstance(d, dict) else None
    if stored and stored != h:
        raise ValueError(f"{p}: stored config_hash {stored} does not match its configuration ({h}); "
                         "the frozen configuration was edited after the freeze gate")
    return cfg, h, True


def frozen_record(path: str | Path | None = None) -> dict[str, Any] | None:
    p = Path(path) if path is not None else FROZEN
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d if isinstance(d, dict) else None


def llm_freeze_meta(client: Any = None, path: str | Path | None = None, model: str | None = None) -> dict[str, Any]:
    """Prompt/model part of the freeze gate for an LLM result: current per-role prompt hashes and
    whether they (and the client's model) match the frozen record."""
    from ..llm.prompts import changed_roles, prompt_hashes

    cur = prompt_hashes()
    model = model or getattr(client, "model", None)
    rec = frozen_record(path)
    llm = (rec or {}).get("llm") or {}
    out: dict[str, Any] = {"prompt_hashes": cur, "model": model}
    if not llm.get("prompt_hashes"):
        out["prompts_frozen"] = False
        out["prompts_note"] = ("the freeze record has no prompt hashes: run `python -m campusflex freeze"
                               + (" --force" if rec else "") + "` before held-out LLM runs")
        return out
    notes = []
    changed = changed_roles(cur, llm["prompt_hashes"])
    if changed:
        notes.append(f"prompts changed after the freeze gate: {', '.join(changed)}")
    if model is not None and llm.get("model") and model != llm["model"]:
        notes.append(f"model {model} differs from the frozen model {llm['model']}")
    out["prompts_frozen"] = not notes
    if notes:
        out["prompts_note"] = "; ".join(notes)
    return out


def notices_freeze_meta(path: str | Path | None = None) -> dict[str, Any]:
    """Held-out notice part of the freeze gate (E3 and the notice scenarios)."""
    from ..llm.prompts import notices_hash

    cur = notices_hash()
    stored = (frozen_record(path) or {}).get("notices_hash")
    out: dict[str, Any] = {"notices_hash": cur, "notices_frozen": bool(stored) and stored == cur}
    if not out["notices_frozen"]:
        out["notices_note"] = ("held-out notices differ from the freeze gate" if stored else
                               "the freeze record has no notices hash: run `python -m campusflex freeze --force`")
    return out


def freeze_meta(h: str, frozen: bool) -> dict[str, Any]:
    return {"config_hash": h, "frozen": frozen,
            **({} if frozen else {"frozen_note": "no results/frozen_config.json: declared defaults used; "
                                                 "run `python -m campusflex freeze` before held-out runs"})}


# ------------------------------------------------------------------------- LLM arms
MAX_FALLBACK_RATE = 0.5      # an LLM arm with more fallbacks/failed calls than this is not an LLM result


def llm_client(cache: bool = True, provider: str | None = None, model: str | None = None) -> tuple[Any, str | None]:
    """(client, None) after one real preflight call, else (None, reason). Every LLM arm uses this."""
    from ..llm.client import live_client_or_reason

    return live_client_or_reason(cache=cache, kind=provider, model=model)


def not_run(reason: str | None) -> dict[str, Any]:
    return {"status": "not_run", "reason": reason or "no LLM client"}


def first_account_error(errors: Any) -> str | None:
    """The first account error (HTTP 401/402) among error strings, or None."""
    from ..llm.client import is_account_error

    return next((str(e) for e in errors if is_account_error(e)), None)


def llm_arm_stats(n_calls: int, n_failed_calls: int, n_fallbacks: int | None = None,
                  n_proposals: int | None = None, n_valid: int | None = None,
                  max_rate: float | None = None, account_error: str | None = None) -> dict[str, Any]:
    """Per-arm LLM health. ``status`` is "run" only when the LLM actually answered: zero calls
    -> "invalid_no_llm_calls"; fallback or failed-call rate above ``MAX_FALLBACK_RATE`` ->
    "invalid_mostly_fallback" (such an arm is a masks/rule result and must not be compared).
    Any call failing with an account error (HTTP 401/402: key or balance) -> "aborted": the
    failures say nothing about the model, so the arm is not reported as a model result."""
    max_rate = MAX_FALLBACK_RATE if max_rate is None else max_rate
    n_fb = n_failed_calls if n_fallbacks is None else n_fallbacks
    fb_rate = (n_fb / n_calls) if n_calls else None
    fail_rate = (n_failed_calls / n_calls) if n_calls else None
    out: dict[str, Any] = {
        "n_calls": int(n_calls),
        "n_failed_calls": int(n_failed_calls),
        "n_fallbacks": int(n_fb),
        "failed_call_rate": fail_rate,
        "fallback_rate": fb_rate,
        "n_proposals": None if n_proposals is None else int(n_proposals),
        "n_valid": None if n_valid is None else int(n_valid),
        "valid_proposal_rate": (n_valid / n_proposals) if (n_proposals and n_valid is not None) else None,
        "max_fallback_rate": max_rate,
    }
    if account_error:
        out["account_error"] = str(account_error)[:200]
        out["status"] = "aborted"
        out["reason"] = (f"LLM account error: {str(account_error)[:160]}. Not a model result; fix the account "
                         "(top up / key) and rerun.")
    elif not n_calls:
        out["status"], out["reason"] = "invalid_no_llm_calls", "the LLM was never called"
    elif max(fb_rate or 0.0, fail_rate or 0.0) > max_rate:
        out["status"] = "invalid_mostly_fallback"
        out["reason"] = (f"fallback rate {fb_rate:.2f} / failed-call rate {fail_rate:.2f} > {max_rate}: "
                         "results are mostly the non-LLM fallback")
    else:
        out["status"] = "run"
    return out


def llm_stats_by_arm(rows: list[dict[str, Any]], arm_key: str = "arm") -> dict[str, dict[str, Any]]:
    """Pooled L2 statistics per LLM arm from E1 C/D episode rows."""
    out: dict[str, dict[str, Any]] = {}
    for arm in sorted({r.get(arm_key) for r in rows if r.get(arm_key) is not None}):
        rs = [r for r in rows if r.get(arm_key) == arm]

        def g(k: str, rs: list[dict[str, Any]] = rs) -> int:
            return int(sum(int(r.get(k) or 0) for r in rs))

        calls, fb = g("llm_l2_calls"), g("llm_l2_fallbacks")
        failed = g("llm_n_failed_calls") if any("llm_n_failed_calls" in r for r in rs) else fb
        st = llm_arm_stats(calls, failed, fb, g("llm_l2_plans"), g("llm_l2_valid"))
        sel = [r["llm_selection_rate"] for r in rs if r.get("llm_selection_rate") is not None]
        st["llm_selection_rate"] = float(np.mean(sel)) if sel else None
        st["n_empty_replies"] = g("llm_l2_empty")        # valid "no staged plan" answers (not fallbacks)
        st["empty_reply_rate"] = (st["n_empty_replies"] / calls) if calls else None
        st["episodes"] = len(rs)
        out[str(arm)] = st
    return out


# ----------------------------------------------------------- Eq. 4 order in comparisons
def lexicographic_compare(u_a: list[float], u_b: list[float], j_a: list[float], j_b: list[float],
                          n_boot: int = 10_000, seed: int = 0) -> dict[str, Any]:
    """Paired comparison of arm a against arm b in the order of Equation 4: unserved
    person-minutes U first, then the weighted score J. U decides when its paired interval
    excludes zero; when U is equal in every pair, or its interval includes zero (no detectable
    difference in service), the score J decides (``decided_by`` says which). Top-level fields are
    the score (J) bootstrap for display; ``decided_by`` / ``a_better`` / ``a_worse`` carry the verdict."""
    bu = paired_bootstrap(u_a, u_b, n_boot=n_boot, seed=seed)
    bj = paired_bootstrap(j_a, j_b, n_boot=n_boot, seed=seed)
    du = np.asarray(u_a, float) - np.asarray(u_b, float)
    if np.all(np.abs(du) < 1e-9):
        decided, better, worse = "score", bj["hi"] < 0, bj["lo"] > 0
    elif bu["excludes_zero"]:
        decided, better, worse = "unserved", bu["hi"] < 0, bu["lo"] > 0
    else:
        decided, better, worse = "score_unserved_tied", bj["hi"] < 0, bj["lo"] > 0
    return {**bj, "score": bj, "unserved": bu, "decided_by": decided, "a_better": bool(better), "a_worse": bool(worse)}


def compare_arms(rows: list[dict[str, Any]], arm_key: str, a: str, b: str,
                 pair_keys: tuple[str, ...] = ("scenario", "seed", "mismatch"), **kw: Any) -> dict[str, Any] | None:
    """Lexicographic paired comparison; repeats of an arm are averaged per pair before resampling."""
    ua, ub, keys = pair_rows(rows, arm_key, a, b, pair_keys, metric="unserved_person_min")
    ja, jb, keys2 = pair_rows(rows, arm_key, a, b, pair_keys, metric="score")
    if not keys or keys != keys2:
        return None
    return {**lexicographic_compare(ua, ub, ja, jb, **kw), "keys": [list(k) for k in keys]}


def config_hashes(rows: list[dict[str, Any]], arm_key: str, default: dict[str, str | None] | None = None) -> dict[str, list]:
    """Distinct config hashes per arm (row tag, else the file-level default for that arm)."""
    out: dict[str, set] = {}
    for r in rows:
        arm = r.get(arm_key)
        h = r.get("config_hash") or (default or {}).get(str(arm))
        out.setdefault(str(arm), set())
        if h:
            out[str(arm)].add(h)
    return {k: sorted(v) for k, v in out.items()}


def config_conflict(hashes: dict[str, list], a: str, b: str) -> str | None:
    """Reason to refuse comparing arms a and b, or None (unknown hashes are not a conflict)."""
    ha, hb = set(hashes.get(a, [])), set(hashes.get(b, []))
    if len(ha) > 1 or len(hb) > 1:
        return f"rows of one arm come from several configurations: {a}={sorted(ha)}, {b}={sorted(hb)}"
    if ha and hb and ha != hb:
        return f"config_hash differs: {a}={sorted(ha)[0]}, {b}={sorted(hb)[0]}"
    return None
