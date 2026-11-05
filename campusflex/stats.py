"""Paired bootstrap statistics (Sections 4.1, 4.4): comparisons are paired by seed and
condition; intervals use 10,000 resamples by default."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Iterable

import numpy as np


def paired_bootstrap(
    a: Iterable[float], b: Iterable[float], n_boot: int = 10_000, seed: int = 0, ci: float = 0.95
) -> dict[str, Any]:
    """Mean of (a − b) with a percentile bootstrap interval over pairs."""
    a, b = np.asarray(list(a), float), np.asarray(list(b), float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("paired samples must be non-empty and the same length")
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {
        "n": int(d.size),
        "mean_diff": float(d.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def pair_rows(
    rows: list[dict[str, Any]],
    arm_key: str,
    arm_a: str,
    arm_b: str,
    pair_keys: tuple[str, ...] = ("scenario", "seed", "mismatch"),
    reduce: Callable[[list[float]], float] = lambda xs: float(np.mean(xs)),
    metric: str = "score",
) -> tuple[list[float], list[float], list[tuple]]:
    """Align two arms by pair keys; repeats of an arm (LLM arms) are averaged per pair."""
    ga: dict[tuple, list[float]] = defaultdict(list)
    gb: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        k = tuple(r[x] for x in pair_keys)
        if r[arm_key] == arm_a:
            ga[k].append(float(r[metric]))
        elif r[arm_key] == arm_b:
            gb[k].append(float(r[metric]))
    keys = sorted(set(ga) & set(gb))
    return [reduce(ga[k]) for k in keys], [reduce(gb[k]) for k in keys], keys


def group_means(rows: list[dict[str, Any]], by: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict[str, Any]]:
    g: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in by)].append(r)
    out = []
    for k in sorted(g, key=lambda x: tuple(str(v) for v in x)):
        rs = g[k]
        row = dict(zip(by, k))
        row["n"] = len(rs)
        for m in metrics:
            vals = [r[m] for r in rs if r.get(m) is not None]
            row[m] = float(np.mean(vals)) if vals else None
        out.append(row)
    return out
