"""Outcome space of a bilateral multi-issue negotiation.

Own implementation instead of NegMAS (DECISIONS D-005): additive utilities, exact Pareto
frontier by enumeration/sampling, Nash and Kalai–Smorodinsky points, counterpart families and
the alternating-offers protocol. Everything is deterministic given a seed.

The time-dependent tactic has the form given in Baarslag's survey (§3.6), which reproduces
Faratin, Sierra and Jennings:

    u(t) = Pmin + (Pmax - Pmin) * (1 - F(t)),    F(t) = k + (1 - k) * t**(1/e)

with e < 1 Boulware (concedes at the end) and e >= 1 Conceder (concedes quickly).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

IssueType = Literal["continuous", "discrete"]
Offer = dict[str, float | str]


@dataclass(frozen=True)
class Issue:
    issue_id: str
    type: IssueType
    values: tuple = ()          # discrete: ordered options
    low: float = 0.0            # continuous
    high: float = 1.0
    steps: int = 11             # discretisation used to enumerate the space

    def grid(self) -> list:
        if self.type == "discrete":
            return list(self.values)
        return list(np.linspace(self.low, self.high, self.steps))


@dataclass(frozen=True)
class Domain:
    domain_id: str
    issues: tuple[Issue, ...]

    @property
    def issue_ids(self) -> list[str]:
        return [i.issue_id for i in self.issues]

    def outcome_space(self, cap: int = 20000) -> list[Offer]:
        """Enumerate the outcome space; beyond `cap`, sample deterministically."""
        grids = [i.grid() for i in self.issues]
        total = int(np.prod([len(g) for g in grids]))
        if total <= cap:
            return [dict(zip(self.issue_ids, combo)) for combo in itertools.product(*grids)]
        rng = np.random.default_rng(0)
        out: list[Offer] = []
        for _ in range(cap):
            out.append({i.issue_id: g[rng.integers(len(g))] for i, g in zip(self.issues, grids)})
        return out


@dataclass
class Utility:
    """Additive utility: u(o) = sum_i w_i * v_i(o_i), with w normalised and v_i in [0,1]."""

    domain: Domain
    weights: dict[str, float]
    value_maps: dict[str, dict | tuple]  # discrete: {value: score}; continuous: (worst, best)
    reservation_value: float = 0.0       # on the utility scale [0,1]

    def __post_init__(self) -> None:
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("weights must sum to more than zero")
        self.weights = {k: v / total for k, v in self.weights.items()}

    def value(self, issue_id: str, raw) -> float:
        vm = self.value_maps[issue_id]
        if isinstance(vm, dict):
            return float(vm[raw])
        worst, best = vm
        if best == worst:
            return 0.0
        return float(np.clip((raw - worst) / (best - worst), 0.0, 1.0))

    def __call__(self, offer: Offer) -> float:
        return float(sum(self.weights[i] * self.value(i, offer[i]) for i in self.weights))

    def weight_vector(self) -> np.ndarray:
        return np.array([self.weights[i] for i in self.domain.issue_ids])


def pareto_frontier(offers: list[Offer], ua: Utility, ub: Utility) -> list[Offer]:
    pts = np.array([[ua(o), ub(o)] for o in offers])
    keep: list[Offer] = []
    for idx, (a, b) in enumerate(pts):
        dominated = np.any((pts[:, 0] >= a) & (pts[:, 1] >= b)
                           & ((pts[:, 0] > a) | (pts[:, 1] > b)))
        if not dominated:
            keep.append(offers[idx])
    return keep


def nash_point(offers: list[Offer], ua: Utility, ub: Utility) -> tuple[Offer, float]:
    """Maximise the product of surpluses over the reservation values."""
    best, best_val = offers[0], -np.inf
    for offer in offers:
        val = max(ua(offer) - ua.reservation_value, 0.0) * max(ub(offer) - ub.reservation_value, 0.0)
        if val > best_val:
            best, best_val = offer, val
    return best, best_val


def kalai_smorodinsky(offers: list[Offer], ua: Utility, ub: Utility) -> Offer:
    """The point where gains relative to the maximal aspiration are equalised."""
    va = np.array([ua(o) for o in offers])
    vb = np.array([ub(o) for o in offers])
    feasible = (va >= ua.reservation_value) & (vb >= ub.reservation_value)
    if not feasible.any():
        feasible = np.ones(len(offers), dtype=bool)
    ideal_a, ideal_b = va[feasible].max(), vb[feasible].max()
    ra = (va - ua.reservation_value) / max(ideal_a - ua.reservation_value, 1e-9)
    rb = (vb - ub.reservation_value) / max(ideal_b - ub.reservation_value, 1e-9)
    score = np.where(feasible, np.minimum(ra, rb) - 0.001 * np.abs(ra - rb), -np.inf)
    return offers[int(np.argmax(score))]


def zopa(offers: list[Offer], ua: Utility, ub: Utility) -> list[Offer]:
    return [o for o in offers
            if ua(o) >= ua.reservation_value and ub(o) >= ub.reservation_value]


# --------------------------------------------------------------------------- tactics

def target_utility(t: float, e: float, k: float = 0.0,
                   p_min: float = 0.0, p_max: float = 1.0) -> float:
    """u(t) from Baarslag's survey §3.6 (Faratin, Sierra and Jennings)."""
    t = float(np.clip(t, 0.0, 1.0))
    f = k + (1.0 - k) * (t ** (1.0 / max(e, 1e-6)))
    return float(p_min + (p_max - p_min) * (1.0 - f))


FAMILIES = ("boulware", "conceder", "linear", "hardliner", "tit_for_tat")


@dataclass
class Counterpart:
    """Scripted counterpart: family + hidden parameters θ."""

    utility: Utility
    family: str
    e: float                  # concession exponent
    deadline: int             # rounds until its deadline
    k: float = 0.0
    tft_alpha: float = 1.0    # reciprocity for tit_for_tat
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def target(self, round_idx: int, opponent_concession: float = 0.0) -> float:
        rv = self.utility.reservation_value
        t = min(round_idx / max(self.deadline, 1), 1.0)
        if self.family == "hardliner":
            return 1.0
        if self.family == "tit_for_tat":
            base = target_utility(t, 1.0, self.k, p_min=rv, p_max=1.0)
            return float(np.clip(base - self.tft_alpha * opponent_concession, rv, 1.0))
        return target_utility(t, self.e, self.k, p_min=rv, p_max=1.0)

    def choose_offer(self, space: list[Offer], target: float,
                     own_values: np.ndarray) -> Offer:
        """The offer whose own utility is closest to the target (stable random tie-break)."""
        idx = np.argsort(np.abs(own_values - target))[:5]
        return space[int(self.rng.choice(idx))]

    def accepts(self, offer: Offer, round_idx: int, opponent_concession: float = 0.0) -> bool:
        u = self.utility(offer)
        return u >= max(self.utility.reservation_value, self.target(round_idx, opponent_concession) - 1e-9)


def theta_of(counterpart: Counterpart) -> dict:
    """The counterpart's true θ, as the foxes estimate it."""
    return {
        "rv": counterpart.utility.reservation_value,
        "beta": counterpart.e,
        "T": float(counterpart.deadline),
        "w": counterpart.utility.weight_vector().tolist(),
        "family": counterpart.family,
    }
