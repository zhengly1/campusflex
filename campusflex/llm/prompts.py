"""Prompt fingerprints for the freeze gate (Section 4.2, Figure 4).

The gate locks "prompts, weight levels, λΩ and held-out notices". ``PlannerConfig`` covers the
weights and λΩ (``config_hash``); this module covers the rest:

* :func:`prompt_hashes` – one hash per LLM role over everything that shapes what the model is
  sent: the system prompt, the output schema(s), the source of the user-prompt builder and of the
  helpers that render its data (``Snapshot.summary``, L4 option views, E4 ranking view, L5 digest), the
  shared JSON-schema instruction and data-field delimiting, and the generation settings
  (per-role ``max_tokens``, temperature). Any edit to an L1–L5 prompt changes that role's hash.
* :func:`notices_hash` – the held-out notice corpus that E3 and the notice scenarios read.

Held-out LLM results whose hashes differ from the frozen record are labelled
``prompts_frozen = false`` and shown as Partial in the evidence ledger.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

NOTICE_CORPUS = Path(__file__).resolve().parents[2] / "data" / "notice_corpus.json"


def _src(obj: Any) -> str:
    return inspect.getsource(obj)


def _digest(parts: list[Any]) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def role_prompt_parts() -> dict[str, list[Any]]:
    """The prompt material of each role (imported lazily: the roles import the LLM client)."""
    from ..experiments import e4
    from ..roles import l1_intent as l1
    from ..roles import l2_proposer as l2
    from ..roles import l3_events as l3
    from ..roles import l4_negotiator as l4
    from ..roles import l5_reviewer as l5

    from ..state import Snapshot

    # Each role: system prompt, schema(s), the user-prompt builder and the helpers that render the
    # data it embeds (state summary, option and ranking views, review digest), so an edit to any
    # of them changes what the model receives and therefore the role's hash.
    return {
        "L1": [l1.L1_SYSTEM, l1.REQUEST_SCHEMA, _src(l1.IntentCompiler.build_prompt)],
        "L2": [l2.L2_SYSTEM, l2.L2_SCHEMA, _src(l2.LLMStagedProposer.build_prompt), _src(Snapshot.summary)],
        "L3": [l3.L3_SYSTEM, l3.ADJUSTMENT_SCHEMA, _src(l3.LLMInterpreter.build_prompt)],
        "L4": [l4.L4_SYSTEM, l4.L4_SCHEMA, _src(l4.explain_prompt), _src(l4._alt), _src(l4._first_action_desc),
               l4.WHATIF_SYSTEM, l4.WHATIF_SCHEMA],
        "L4_explain": [e4.EXPLAIN_SYSTEM, e4.EXPLAIN_SCHEMA, _src(e4.explain_user_prompt), list(e4.KEEP),
                       _src(e4.compact_ranking)],
        "L5": [l5.L5_SYSTEM, l5.PROPOSAL_SCHEMA, _src(l5.LLMReviewer.build_prompt), _src(l5.build_digest),
               _src(l5.interval_table), _src(l5._round)],
    }


def _common_parts() -> list[Any]:
    from . import client as c

    return [_src(c.json_instruction), _src(c.wrap_data), c.DEFAULT_MAX_TOKENS,
            inspect.signature(c.DeepSeekJSONClient.__init__).parameters["temperature"].default]


def prompt_hashes() -> dict[str, str]:
    common = _common_parts()
    return {role: _digest([common, parts]) for role, parts in role_prompt_parts().items()}


def notices_hash(path: str | Path | None = None) -> str | None:
    """Hash of the held-out notices (``heldout`` in ``data/notice_corpus.json``)."""
    p = Path(path) if path is not None else NOTICE_CORPUS
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return _digest([d.get("heldout") if isinstance(d, dict) else d])


def changed_roles(current: dict[str, str], stored: dict[str, str] | None) -> list[str]:
    stored = stored or {}
    return sorted(r for r in set(current) | set(stored) if current.get(r) != stored.get(r))
