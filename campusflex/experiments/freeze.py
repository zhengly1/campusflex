"""Freeze gate (Section 4.2): lock the configuration before any held-out run.

    python -m campusflex freeze [--out results/frozen_config.json] [--force]

Writes ``PlannerConfig()`` (the declared defaults, after development-seed tuning has been
merged into them) together with its ``config_hash``, and locks the rest of what the gate covers
(Figure 4: "prompts, weight levels, λΩ and held-out notices"): one hash per LLM role prompt
(L1–L5, see ``campusflex.llm.prompts``), the provider/model, and the held-out notice corpus.
Held-out runs (E1, E3, E4, E6 pre-review
arm and the E6 review base) read this file via ``common.frozen_config()``; without it they still
run but are labelled ``meta.frozen = false`` and the evidence ledger shows them as Partial.
The gate is passed once: an existing file is only replaced with ``--force`` (e.g. after a teammate
rewrote the held-out notices, which must happen before any held-out run).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import PlannerConfig
from . import common


def freeze(out: str | Path | None = None, force: bool = False, cfg: PlannerConfig | None = None,
           source: str = "PlannerConfig() defaults") -> dict[str, Any]:
    p = Path(out) if out is not None else common.FROZEN
    if not p.is_absolute():
        p = common.ROOT / p
    if p.exists() and not force:
        raise FileExistsError(f"{p} exists: the freeze gate was already passed (use --force to replace it)")
    cfg = cfg or PlannerConfig()
    from ..llm.client import DEFAULT_MODEL, DEFAULT_PROVIDER
    from ..llm.prompts import notices_hash, prompt_hashes

    rec = {"config": cfg.to_dict(), "config_hash": common.config_hash(cfg), "source": source,
           "llm": {"provider": DEFAULT_PROVIDER, "model": DEFAULT_MODEL, "prompt_hashes": prompt_hashes()},
           "notices_hash": notices_hash(), "scenario_hash": common.scenario_hash(),
           "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1, sort_keys=True))
    return {"path": str(p), **rec}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="default: results/frozen_config.json")
    ap.add_argument("--force", action="store_true", help="replace an existing frozen configuration")
    a = ap.parse_args(argv)
    try:
        rec = freeze(a.out, a.force)
    except FileExistsError as e:
        raise SystemExit(str(e)) from None
    print(f"wrote {rec['path']} (config_hash {rec['config_hash']}, model {rec['llm']['model']}, "
          f"notices_hash {rec['notices_hash']}, scenario_hash {rec['scenario_hash']})")


if __name__ == "__main__":
    main()
