"""Rebuild a live LLM episode for the dashboard from the recorded transcripts – no API call.

    python -m campusflex.dashboard.replay_llm_demo [--arm arm_C] [--scenario fault] [--seed 10] [--repeat 0]

``results/e1_CD_traces.jsonl.gz`` keeps, for every L2 call of the live E1 C/D run, the prompt
hash, the model's reply and its latency. They are written into a private prompt-hash cache
and the same episode is run again with a client that cannot reach the network: every L2
prompt must hash to a recorded reply (same prompts, configuration and generation settings as
the live run) and each cached reply carries its original latency, so the step budget decides
as it did live. The replay is saved only when every step commits the same plan from the same
source as the trace (``results/demo_episode_llm.json``, labelled as a replay).
"""
from __future__ import annotations

import argparse
import gzip
import json
import tempfile
from pathlib import Path
from typing import Any

from ..experiments.common import RESULTS, frozen_config
from ..llm.client import CachedClient, DeepSeekJSONClient, LLMResponse

TRACES = RESULTS / "e1_CD_traces.jsonl.gz"
RESPONSE_FIELDS = ("role", "data", "model", "latency_s", "input_tokens", "output_tokens", "stop_reason",
                   "prompt_hash", "request_id", "system_fingerprint", "cache_hit_tokens")


class _Offline(DeepSeekJSONClient):
    """Same generation settings (hence prompt hashes) as the live client; never calls the API."""

    def __init__(self, model: str):
        super().__init__(model=model, client=object())
        self.misses = 0

    def complete_json(self, *, role: str, system: str, user: str, schema: dict[str, Any],
                      effort: str | None = None) -> LLMResponse:
        self.misses += 1
        return LLMResponse(role, None, model=self.model, error="replay_miss: no recorded reply for this prompt")


def load_trace(arm: str, scenario: str, seed: int, repeat: int, mismatch: bool = False,
               path: Path = TRACES) -> dict[str, Any]:
    with gzip.open(path, "rt") as f:
        for line in f:
            ep = json.loads(line)
            if (ep["arm"], ep["scenario"], ep["seed"], ep["repeat"], bool(ep.get("mismatch"))) == \
                    (arm, scenario, seed, repeat, mismatch):
                return ep
    raise SystemExit(f"no trace for {arm} {scenario} seed {seed} repeat {repeat} in {path}")


def replay(trace: dict[str, Any], out: Path | None = None) -> dict[str, Any]:
    from ..episode import run_episode
    from ..policies import make_policy
    from ..roles.l2_proposer import LLMStagedProposer
    from ..scenarios import make_scenario

    calls = [t for d in trace["decisions"] for t in (d.get("llm") or [])
             if t.get("prompt_hash") and t.get("data") is not None and not t.get("error")]
    model = next((t["model"] for t in calls if t.get("model")), "deepseek-flash")
    cfg, h, frozen = frozen_config()
    if trace.get("config_hash") and trace["config_hash"] != h:
        raise SystemExit(f"trace config {trace['config_hash']} differs from the frozen config {h}")
    with tempfile.TemporaryDirectory() as cache:
        for t in calls:
            entry = {k: t.get(k) for k in RESPONSE_FIELDS}
            entry["text"] = ""
            Path(cache, f"{t['prompt_hash']}.json").write_text(json.dumps(entry))
        inner = _Offline(model)
        client = CachedClient(inner, cache_dir=cache, enabled=True)
        pol = make_policy(trace["arm"], cfg, l2=LLMStagedProposer(client, cfg))
        r = run_episode(make_scenario(trace["scenario"], trace["seed"], mismatch=trace.get("mismatch", False)),
                        pol, cfg, keep_decisions=True)
    live = [(d["step"], d["plan_id"], d["source"]) for d in trace["decisions"]]
    again = [(d["step"], d["plan_id"], d["source"]) for d in r.decisions]
    diff = [(a, b) for a, b in zip(live, again) if a != b]
    ok = not diff and len(live) == len(again) and inner.misses == 0
    report = {"verified": ok, "steps": len(again), "recorded_l2_replies": len(calls),
              "replay_misses": inner.misses, "differing_steps": diff[:5],
              "l2_committed_steps": sum(1 for _, _, s in again if s.endswith("L2"))}
    if ok and out is not None:
        r.save(out)
        d = json.loads(Path(out).read_text())
        d["provenance"] = {"kind": "replay of the live E1 C/D run (recorded replies, no API call)",
                           "arm": trace["arm"], "scenario": trace["scenario"], "seed": trace["seed"],
                           "repeat": trace["repeat"], "model": model, "config_hash": h, **report}
        Path(out).write_text(json.dumps(d, indent=1, default=str))
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="arm_C", choices=["arm_C", "arm_D"])
    ap.add_argument("--scenario", default="fault")
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--out", default=str(RESULTS / "demo_episode_llm.json"))
    a = ap.parse_args(argv)
    rep = replay(load_trace(a.arm, a.scenario, a.seed, a.repeat), Path(a.out))
    print(json.dumps(rep, indent=1))
    if not rep["verified"]:
        raise SystemExit("replay differs from the live run; nothing written")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
