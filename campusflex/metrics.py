"""Episode metrics (Section 4.6, Eq. 5, Table 15).

The full-evening combined score is Equation 4 over the whole 18:00–22:00 trace,
excluding Ω. It is always computed with the *reference* (v1) weights so that
configurations with different weights (L1, L5) are compared on the same yardstick;
the score under the run's own weights is reported alongside.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .config import STEP_MIN, PlannerConfig

REFERENCE_WEIGHTS = {"lambda_T": 8.0, "lambda_M": 0.06, "lambda_S": 0.08}


def full_evening_score(E: float, V: float, M: float, K: float, w: dict[str, float] = REFERENCE_WEIGHTS) -> float:
    return E + w["lambda_T"] * V + w["lambda_M"] * M + w["lambda_S"] * K


def episode_metrics(logs: list[dict[str, Any]], cfg: PlannerConfig | None = None) -> dict[str, Any]:
    E = sum(l["E"] for l in logs)
    V = sum(l["V"] for l in logs)
    occ_h = sum(l["occ_h"] for l in logs)
    served = sum(l["served"] for l in logs)
    demand = sum(l["demand"] for l in logs)
    U = sum(l["unserved"] for l in logs) * STEP_MIN
    M = sum(l["M"] for l in logs)
    K = sum(l["K"] for l in logs)
    LS = sum(l["LS"] for l in logs)
    out = {
        "kwh": round(E, 4),
        "service_pct": round(100.0 * served / demand, 4) if demand else 100.0,
        "dev_c": round(V / occ_h, 5) if occ_h else 0.0,       # D_T, Eq. 5
        "V_deg_person_h": round(V, 4),
        "occupied_person_h": round(occ_h, 3),
        "unserved_person_min": int(U),
        "moves": int(M),
        "forced_moves": int(sum(l.get("forced_moves", 0) for l in logs)),
        "loss_of_seat": int(LS),
        "switches": int(K),
        "hard_violations": int(sum(l.get("violations", 0) for l in logs)),
        "score": round(full_evening_score(E, V, M, K), 4),
        "max_occupied_temp_c": round(
            max((max((t for t, n in zip(l["T_end"], l["counts"]) if n > 0), default=0.0) for l in logs), default=0.0), 3
        ),
    }
    if cfg is not None:
        out["score_cfg_weights"] = round(
            full_evening_score(E, V, M, K, {"lambda_T": cfg.lambda_T, "lambda_M": cfg.lambda_M, "lambda_S": cfg.lambda_S}), 4
        )
    # V-gap and LLM selection where available
    pv = [l for l in logs if l.get("pred_first_V") is not None]
    if pv:
        out["vgap_first_interval"] = round(float(np.mean([l["V"] - l["pred_first_V"] for l in pv])), 5)
    src = [l.get("source") for l in logs if l.get("source")]
    if src:
        out["llm_selection_rate"] = round(sum(1 for s in src if s.endswith("L2")) / len(src), 4)
        # staged = a rule-staged or LLM-staged plan (fresh or carried forward); fallback holds are not
        out["staged_selection_rate"] = round(sum(1 for s in src if s.split(":")[-1] in ("rule", "L2")) / len(src), 4)
    return out
