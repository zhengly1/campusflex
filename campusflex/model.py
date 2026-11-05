"""Retained numerical core (Section 3.1): one-zone RC thermal model, energy accounting,
thermostat and the count-level seat-assignment transition shared by the planner's
rollouts and the environment.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import (
    BAND_HIGH_C,
    BAND_LOW_C,
    COP,
    DT_H,
    EFFICIENCY_REF_TOUT_C,
    N_ROOMS,
    Q_OCC_KW,
    ROOMS,
    TARGET_C,
    TRANSFER_C,
)


@dataclass(frozen=True)
class ThermalParams:
    R: np.ndarray
    C: np.ndarray
    P_ac: np.ndarray
    P_light: np.ndarray
    cop: np.ndarray

    @classmethod
    def nominal(cls) -> "ThermalParams":
        return cls(
            R=np.array([r.R for r in ROOMS]),
            C=np.array([r.C for r in ROOMS]),
            P_ac=np.array([r.P_ac for r in ROOMS]),
            P_light=np.array([r.P_light for r in ROOMS]),
            cop=np.full(N_ROOMS, COP),
        )

    @classmethod
    def perturbed(cls, mismatch: dict[str, dict[str, float]] | None) -> "ThermalParams":
        """Environment parameters under declared ±10% model mismatch (Section 4.2)."""
        p = cls.nominal()
        if not mismatch:
            return p
        f = {k: np.array([mismatch[r.room_id][k] for r in ROOMS]) for k in ("R", "C", "COP")}
        return cls(R=p.R * f["R"], C=p.C * f["C"], P_ac=p.P_ac, P_light=p.P_light, cop=p.cop * f["COP"])


NOMINAL = ThermalParams.nominal()
SEATS = np.array([r.seats for r in ROOMS], dtype=int)


def deviation(T: np.ndarray | float, low: float = BAND_LOW_C, high: float = BAND_HIGH_C) -> np.ndarray:
    """δ(T) = max(0, 24 − T) + max(0, T − 27)  (Eq. 5)."""
    T = np.asarray(T, dtype=float)
    return np.maximum(0.0, low - T) + np.maximum(0.0, T - high)


def thermostat_u(
    T: np.ndarray,
    t_out: float,
    n: np.ndarray,
    setpoint: np.ndarray,
    health: np.ndarray,
    params: ThermalParams = NOMINAL,
) -> np.ndarray:
    """Local device thermostat: cooling fraction that drives T to the setpoint in one interval.

    ``setpoint`` is NaN where the AC is off. The controller always uses *nominal* parameters
    (it does not know the true plant); 0 ≤ u ≤ h enforces Eq. (1).
    """
    load = (t_out - T) / params.R + Q_OCC_KW * n
    need = load + (T - np.nan_to_num(setpoint, nan=0.0)) * params.C / DT_H
    u = need / (params.cop * params.P_ac)
    u = np.where(np.isnan(setpoint), 0.0, u)
    return np.clip(u, 0.0, np.maximum(health, 0.0))


def step_temps(T: np.ndarray, t_out: float, n: np.ndarray, u: np.ndarray, params: ThermalParams) -> np.ndarray:
    """Eq. (2): T_{i,t+1} = T_{i,t} + Δt/C_i [ (T_out − T)/R_i + q_occ n − COP_i P_i^AC u ]."""
    return T + DT_H / params.C * ((t_out - T) / params.R + Q_OCC_KW * n - params.cop * params.P_ac * u)


def power_kw(u: np.ndarray, lights: np.ndarray, params: ThermalParams = NOMINAL) -> np.ndarray:
    """Per-room electrical power; Eq. (3) sums Δt × this over steps."""
    return params.P_ac * u + params.P_light * lights


def efficiency_rank(t_out_ref: float = EFFICIENCY_REF_TOUT_C) -> list[int]:
    """Rooms ordered by predicted kWh per seat-hour at full occupancy (efficiency-ranked threshold)."""
    p = NOMINAL
    cool_kw = ((t_out_ref - TARGET_C) / p.R + Q_OCC_KW * SEATS) / p.cop
    per_seat = (cool_kw + p.P_light) / SEATS
    return [int(i) for i in np.argsort(per_seat, kind="stable")]


EFFICIENCY_ORDER = efficiency_rank()
INVENTORY_ORDER = list(range(N_ROOMS))


def smart_fill_order(temps: np.ndarray, health: np.ndarray, high: float = BAND_HIGH_C) -> list[int]:
    """Seat-assignment preference: working AC first, then rooms within the band, then efficiency."""
    rank = {i: k for k, i in enumerate(EFFICIENCY_ORDER)}
    return sorted(range(N_ROOMS), key=lambda i: (health[i] <= 0.0, temps[i] > high, rank[i]))


@dataclass
class AssignmentResult:
    counts: np.ndarray                  # occupancy n_{i,t} during the interval
    moved: int = 0                      # relocations of continuing students (forced + explicit)
    forced: int = 0
    loss_of_seat: int = 0               # seated students who lost their seat
    unserved: int = 0                   # d_t − Σ n_{i,t}
    switches: int = 0
    pairs: list[tuple[int, int, int]] = field(default_factory=list)   # (from, to, k) relocations
    seated_new: list[tuple[int, int]] = field(default_factory=list)  # (room, k) waiting → seat


def _largest_remainder_remove(counts: np.ndarray, k: int) -> np.ndarray:
    """Remove k students proportionally (deterministic) – used for departures in rollouts."""
    total = int(counts.sum())
    if k <= 0 or total == 0:
        return counts.copy()
    k = min(k, total)
    share = counts * (k / total)
    take = np.floor(share).astype(int)
    rem = k - int(take.sum())
    order = np.argsort(-(share - take), kind="stable")
    for i in order[:rem]:
        take[i] += 1
    return counts - take


def assign(
    counts: np.ndarray,
    open_prev: np.ndarray,
    open_new: np.ndarray,
    capacity: np.ndarray,
    demand: int,
    temps: np.ndarray,
    fill_order: list[int],
    moves: list[tuple[int, int, int]] = (),
    depart_proportional: bool = True,
) -> AssignmentResult:
    """Count-level transition applied at the start of an interval.

    Order: departures → explicit moves (count for "all" fixed before any move) → forced
    relocation of whoever remains in an unusable room → seat waiting students. Departing students are never counted as moves. A room is usable if it
    is open, has capacity and is below the transfer threshold. ``moves`` uses count = -1 for "all".
    Invariants (Eq. 1): 0 ≤ n_i ≤ c_i o_i and Σ n_i ≤ d.
    """
    n = counts.astype(int).copy()
    if depart_proportional and n.sum() > demand:
        n = _largest_remainder_remove(n, int(n.sum()) - demand)
    usable = open_new.astype(bool) & (capacity > 0) & (temps < TRANSFER_C)
    res = AssignmentResult(counts=n, switches=int(np.sum(open_prev.astype(bool) != open_new.astype(bool))))

    def free(i: int) -> int:
        return int(capacity[i] - n[i]) if usable[i] else 0

    # explicit moves first, so a transfer out of a closing room reaches its named target;
    # "all" means the occupants before any move this step (nobody is moved twice)
    base = n.copy()
    moved_in = np.zeros(len(n), dtype=int)
    for f, t, c in moves:
        if not usable[t] or f == t:
            continue
        stay = int(n[f]) - int(moved_in[f])          # students who were not just moved into f
        want = min(int(base[f]) if c < 0 else int(c), stay)
        k = min(want, free(t))
        if k > 0:
            n[f] -= k
            n[t] += k
            moved_in[t] += k
            res.moved += k
            res.pairs.append((f, t, k))
    # forced relocation of whoever is left in an unusable room
    for i in range(len(n)):
        if n[i] > 0 and not usable[i]:
            for j in fill_order:
                if n[i] == 0:
                    break
                if j == i:
                    continue
                k = min(int(n[i]), free(j))
                if k > 0:
                    n[i] -= k
                    n[j] += k
                    res.moved += k
                    res.forced += k
                    res.pairs.append((i, j, k))
            if n[i] > 0:          # nowhere to go: loss of seat
                res.loss_of_seat += int(n[i])
                n[i] = 0
    # seat waiting students (arrivals, or students who lost a seat)
    waiting = max(0, demand - int(n.sum()))
    for j in fill_order:
        if waiting == 0:
            break
        k = min(waiting, free(j))
        if k > 0:
            n[j] += k
            waiting -= k
            res.seated_new.append((j, k))
    res.counts = n
    res.unserved = max(0, demand - int(n.sum()))
    return res
