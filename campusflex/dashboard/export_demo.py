"""Write an episode JSON for the dashboard.

    python -m campusflex.dashboard.export_demo [--policy arm_B] [--scenario fault] [--seed 10]

Runs one non-LLM episode with full decision records and saves ``results/demo_episode.json``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..config import PlannerConfig
from ..episode import run_episode
from ..experiments.common import RESULTS
from ..policies import make_policy
from ..scenarios import make_scenario


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default="arm_B")
    ap.add_argument("--scenario", default="fault")
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--mismatch", action="store_true")
    ap.add_argument("--out", default=str(RESULTS / "demo_episode.json"))
    a = ap.parse_args(argv)
    cfg = PlannerConfig()
    sc = make_scenario(a.scenario, a.seed, mismatch=a.mismatch)
    res = run_episode(sc, make_policy(a.policy, cfg), cfg, keep_decisions=True)
    res.save(a.out)
    print(f"wrote {a.out}: {res.metrics}")
    return Path(a.out)


if __name__ == "__main__":
    main()
