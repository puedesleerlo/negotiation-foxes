"""Common fox interface (§6.2 of the specification).

Every fox returns a `Posterior`: weighted samples over the components of θ it estimates, plus a
summary (mean, quantiles 5/25/50/75/95, entropy) and a scope verification. Never a point.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np

from foxes.domain import Domain, Offer, Utility

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


@dataclass
class FoxMeta:
    fox_id: str
    version: str
    estimates: list[str]
    phase: str                      # prep | online | both
    inputs_required: list[str]
    scope_conditions: list[str]
    assumptions: list[str]
    paper_ids: list[str]
    calibration_dataset: Optional[str] = None
    validation_report: Optional[str] = None
    status: str = "candidate"


@dataclass
class Observation:
    """What the table reveals in one round. Never contains sealed truth.

    `action == "propose"`: `offer` is the counterpart's proposal.
    `action in ("accept", "reject")`: `offer` is *this party's* offer being responded to;
    `party` is who responded. Foxes that model proposals must not confuse the two — see
    `counterpart_offers`.
    """

    round: int
    party: str                       # who acted
    offer: Optional[Offer] = None
    action: str = "propose"          # propose | accept | reject | walk_away | message_only
    utility_to_observer: Optional[float] = None      # what the offer is worth to the observer
    est_utility_to_proposer: Optional[float] = None  # estimate of its worth to the proposer
    max_rounds: Optional[int] = None


@dataclass
class BeliefState:
    """A fox's internal state for *one* party. Foxes never share state across parties."""

    domain: Domain
    own_utility: Optional[Utility] = None
    observations: list[Observation] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    program_id: Optional[str] = None
    party: Optional[str] = None
    round: int = 0


@dataclass
class Posterior:
    """Weighted samples over the estimated parameters."""

    fox_id: str
    version: str
    params: list[str]
    samples: np.ndarray                    # (n, dim)
    weights: Optional[np.ndarray] = None   # (n,), unnormalised
    scope_ok: bool = True
    warnings: list[str] = field(default_factory=list)
    program_id: Optional[str] = None
    party: Optional[str] = None
    step: Optional[str] = None
    round: Optional[int] = None

    def __post_init__(self) -> None:
        self.samples = np.atleast_2d(np.asarray(self.samples, dtype=float))
        if self.samples.shape[1] != len(self.params) and self.samples.shape[0] == len(self.params):
            self.samples = self.samples.T
        if self.weights is None:
            self.weights = np.ones(len(self.samples))
        self.weights = np.asarray(self.weights, dtype=float)
        total = self.weights.sum()
        if total <= 0 or not np.isfinite(total):
            self.weights = np.ones(len(self.samples))
            self.warnings.append("degenerate weights: replaced by uniform weights")
        self.weights = self.weights / self.weights.sum()

    # ------------------------------------------------------------------ summary
    def column(self, param: str) -> np.ndarray:
        return self.samples[:, self.params.index(param)]

    def mean(self, param: str) -> float:
        return float(np.sum(self.weights * self.column(param)))

    def quantile(self, param: str, q: float) -> float:
        x = self.column(param)
        order = np.argsort(x)
        cw = np.cumsum(self.weights[order])
        return float(np.interp(q, cw, x[order]))

    def interval(self, param: str, level: float = 0.9) -> tuple[float, float]:
        lo = (1.0 - level) / 2.0
        return self.quantile(param, lo), self.quantile(param, 1.0 - lo)

    def effective_sample_size(self) -> float:
        return float(1.0 / np.sum(self.weights ** 2))

    def entropy(self, param: str, bins: int = 40) -> float:
        """Differential entropy estimated by a weighted histogram (nats).

        Biased but stable and comparable across rounds, which is what it is used for: to follow
        the *trajectory* of uncertainty, not its absolute value.
        """
        x = self.column(param)
        if np.allclose(x, x[0]):
            return float("-inf")
        hist, edges = np.histogram(x, bins=bins, weights=self.weights, density=False)
        width = edges[1] - edges[0]
        p = hist / max(hist.sum(), 1e-12)
        nz = p > 0
        return float(-np.sum(p[nz] * np.log(p[nz] / width)))

    def summary(self) -> dict:
        out: dict[str, Any] = {"n_samples": int(len(self.samples)),
                               "ess": self.effective_sample_size(),
                               "scope_ok": self.scope_ok,
                               "warnings": list(self.warnings)}
        for param in self.params:
            out[param] = {
                "mean": self.mean(param),
                **{f"q{int(q * 100):02d}": self.quantile(param, q) for q in QUANTILES},
                "entropy": self.entropy(param),
            }
        return out

    def resample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        idx = rng.choice(len(self.samples), size=n, p=self.weights)
        return self.samples[idx]


@dataclass
class ProbEstimate:
    """A probability with an interval: foxes never return a bare number."""

    value: float
    low: float
    high: float
    fox_id: str
    version: str
    scope_ok: bool = True
    warnings: list[str] = field(default_factory=list)


@dataclass
class Diagnostics:
    entropy: dict[str, float]
    scope_ok: bool
    n_observations: int
    warnings: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Fox(Protocol):
    meta: FoxMeta

    def init_state(self, domain: Domain, own_utility: Optional[Utility] = None,
                   **kwargs: Any) -> BeliefState: ...
    def prior(self, state: BeliefState) -> Posterior: ...
    def update(self, state: BeliefState, obs: Observation) -> BeliefState: ...
    def posterior(self, state: BeliefState) -> Posterior: ...
    def diagnostics(self, state: BeliefState) -> Diagnostics: ...


class BaseFox:
    """Shared behaviour; concrete foxes override what is theirs."""

    meta: FoxMeta

    def init_state(self, domain: Domain, own_utility: Optional[Utility] = None,
                   **kwargs: Any) -> BeliefState:
        state = BeliefState(domain=domain, own_utility=own_utility)
        state.data.update(kwargs)
        return state

    def update(self, state: BeliefState, obs: Observation) -> BeliefState:
        state.observations.append(obs)
        state.round = max(state.round, obs.round)
        return state

    def prior(self, state: BeliefState) -> Posterior:
        return self.posterior(state)

    def posterior(self, state: BeliefState) -> Posterior:  # pragma: no cover - abstract
        raise NotImplementedError

    def p_accept(self, state: BeliefState, offer: Offer) -> ProbEstimate:
        raise NotImplementedError(f"{self.meta.fox_id} does not estimate acceptance probability")

    def scope_check(self, state: BeliefState) -> tuple[bool, list[str]]:
        return True, []

    def diagnostics(self, state: BeliefState) -> Diagnostics:
        post = self.posterior(state)
        ok, warns = self.scope_check(state)
        return Diagnostics(
            entropy={p: post.entropy(p) for p in post.params},
            scope_ok=ok, n_observations=len(state.observations), warnings=warns,
            extra={"ess": post.effective_sample_size()},
        )


def counterpart_offers(state: BeliefState, party: str | None = None) -> list[Observation]:
    """The counterpart's *proposals*, in order.

    Responses (`accept`/`reject`) carry this party's own offer and must not be mistaken for
    proposals by the counterpart: a fox that models the concession curve would otherwise read
    our own bids as theirs.
    """
    who = party or state.data.get("counterpart_party")
    return [o for o in state.observations
            if o.offer is not None and o.action == "propose"
            and (who is None or o.party == who)]
