"""Command-line entry point: ``python -m campusflex <command> [args...]``.

Commands (each forwards its arguments to the module's own ``main``):

  run          one episode   (--scenario fault --seed 0 --policy arm_B --device direct|bus|mqtt --save path)
  v1           ledger item 0: reconstructed v1 experiment, 160 episodes
  vgap         ledger item 1: predicted-vs-realized V log + Ω ablation (dev seeds)
  e5           ledger item 3: E5 scripted robustness suite (--live for the LLM sub-test)
  freeze       freeze gate: lock config, prompts (L1–L5 hashes), model and held-out notices
  llm-check    after setting DEEPSEEK_API_KEY: per-role latency/tokens/usability + E1 C/D projection
  e1           ledger items 4/5: E1 arms (--arms AB|CD|all, --reduced)
  t1           ledger item 6: integrated trace T1 (--rehearsal without an LLM)
  e2 e3 e4     ledger item 7
  e6           ledger item 8 (review --kind random|llm, evaluate)
  fixtures     (re)write tests/e5_fixtures
  tables       regenerate every Section 4 table from results/
  ledger       evidence ledger (Table 16) from the files on disk
  demo         export results/demo_episode.json for the dashboard
  smart        smart device extension: per-device IoT dispatch, learned usage profiles, sensor control
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys

COMMANDS = {
    "v1": "campusflex.experiments.v1",
    "vgap": "campusflex.experiments.vgap",
    "freeze": "campusflex.experiments.freeze",
    "llm-check": "campusflex.experiments.llm_check",
    "e1": "campusflex.experiments.e1",
    "e2": "campusflex.experiments.e2",
    "e3": "campusflex.experiments.e3",
    "e4": "campusflex.experiments.e4",
    "e5": "campusflex.experiments.e5",
    "e6": "campusflex.experiments.e6",
    "t1": "campusflex.experiments.trace_t1",
    "fixtures": "campusflex.experiments.e5_fixtures",
    "tables": "campusflex.experiments.tables",
    "ledger": "campusflex.experiments.evidence_ledger",
    "demo": "campusflex.dashboard.export_demo",
    "smart": "campusflex.experiments.smart",
}


def _run_one(argv: list[str]) -> None:
    from .config import PlannerConfig
    from .episode import run_episode
    from .policies import make_policy
    from .scenarios import ALL_SCENARIOS, make_scenario

    ap = argparse.ArgumentParser(prog="campusflex run")
    ap.add_argument("--scenario", default="fault", choices=ALL_SCENARIOS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", default="arm_B",
                    choices=["fixed", "threshold", "threshold_eff", "planner", "planner_nomove", "arm_A", "arm_B"])
    ap.add_argument("--mismatch", action="store_true")
    ap.add_argument("--v1", action="store_true", help="v1 settings (30-min hold, no Ω)")
    ap.add_argument("--device", default="direct", choices=["direct", "bus", "mqtt"])
    ap.add_argument("--save", default=None, help="write the episode JSON (for the dashboard)")
    a = ap.parse_args(argv)
    cfg = PlannerConfig.v1() if a.v1 else PlannerConfig()
    layer = None
    if a.device != "direct":
        from .iot import make_device_layer

        layer = make_device_layer(a.device)
    r = run_episode(make_scenario(a.scenario, a.seed, mismatch=a.mismatch), make_policy(a.policy, cfg), cfg,
                    device_layer=layer, keep_decisions=bool(a.save))
    if a.save:
        r.save(a.save)
    print(json.dumps(r.summary(), indent=1))


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "run":
        _run_one(rest)
        return
    if cmd not in COMMANDS:
        print(f"unknown command {cmd!r}\n{__doc__}")
        sys.exit(2)
    mod = importlib.import_module(COMMANDS[cmd])
    sys.argv = [f"campusflex {cmd}"] + rest
    if hasattr(mod, "main"):
        mod.main()
    else:  # e5_fixtures
        print(mod.write_all())


if __name__ == "__main__":
    main()
