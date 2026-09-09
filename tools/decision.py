"""Decision tools under uncertainty (task 2.8).

Not foxes: they estimate no θ and need no calibration dataset. They take the belief the foxes
produce and turn it into the quantities an offer is decided with.

  t04 `expected_information_gain` — the offer as an experiment (RQ2, exploration lever)
  t05 `evpi`                      — how much resolving a parameter of θ would be worth
  `p_accept_from_belief`          — acceptance probability *derived* from the belief over
                                    (rv, beta, T), instead of estimated separately from the few
                                    data points of one session (f04's v0.2 plan, DECISIONS D-020)

All of them return intervals or distributions, never a bare number.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from foxes.domain import Offer, Utility


@dataclass
class ProbabilityWithInterval:
    value: float
    low: float
    high: float
    n_effective: int
    source: str

    def __repr__(self) -> str:  # pragma: no cover - presentation
        return f"P={self.value:.3f} [{self.low:.3f}, {self.high:.3f}] ({self.source})"


def counterpart_target(rv: np.ndarray, beta: np.ndarray, deadline: np.ndarray,
                       round_idx: int, k: float = 0.0) -> np.ndarray:
    """The counterpart's target utility in the round, per sample of θ.

    Same functional form f03 and the synthetic generator use (Faratin, via Baarslag §3.6): the
    counterpart aims high at first and descends towards its rv as its private deadline nears.
    """
    t = np.clip(round_idx / np.maximum(deadline, 1e-9), 0.0, 1.0)
    f = k + (1.0 - k) * t ** (1.0 / np.maximum(beta, 1e-6))
    return rv + (1.0 - rv) * (1.0 - f)


def p_accept_from_belief(utility_to_counterpart: float, round_idx: int,
                         theta: dict[str, np.ndarray], weights: np.ndarray | None = None,
                         level: float = 0.9) -> ProbabilityWithInterval:
    """P(the counterpart accepts) under the joint belief over (rv, beta, T).

    Not a new model: it is the same belief f01 and f03 already produced, evaluated on the
    question that matters. That is why it needs no acceptance data of its own — which is
    exactly what is not available within one session.
    """
    rv = np.asarray(theta["rv"], dtype=float)
    beta = np.asarray(theta.get("beta", np.ones_like(rv)), dtype=float)
    deadline = np.asarray(theta.get("T", np.full_like(rv, 20.0)), dtype=float)
    w = np.ones_like(rv) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()

    target = counterpart_target(rv, beta, deadline, round_idx)
    accepts = (utility_to_counterpart >= target - 1e-12).astype(float)
    p = float(np.sum(w * accepts))
    n_eff = float(1.0 / np.sum(w ** 2))
    # Wilson interval with the effective sample size: it reflects the Monte Carlo noise of the
    # belief, not an invented extra uncertainty.
    z = 1.959963984540054 if level >= 0.95 else 1.6448536269514722
    denom = 1.0 + z ** 2 / n_eff
    center = (p + z ** 2 / (2 * n_eff)) / denom
    margin = z * np.sqrt(max(p * (1 - p) / n_eff + z ** 2 / (4 * n_eff ** 2), 0.0)) / denom
    return ProbabilityWithInterval(
        value=p, low=float(max(center - margin, 0.0)), high=float(min(center + margin, 1.0)),
        n_effective=int(n_eff), source="belief over (rv, beta, T)")


def _entropy(samples: np.ndarray, weights: np.ndarray, bins: int = 40) -> float:
    if weights.sum() <= 0:
        return 0.0
    w = weights / weights.sum()
    hist, edges = np.histogram(samples, bins=bins, weights=w)
    width = edges[1] - edges[0]
    nz = hist > 0
    if not nz.any():
        return 0.0
    return float(-np.sum(hist[nz] * np.log(hist[nz] / width)))


def expected_information_gain(utility_to_counterpart: float, round_idx: int,
                              theta: dict[str, np.ndarray], weights: np.ndarray | None = None,
                              param: str = "rv") -> dict:
    """t04 — Expected information gain of making *this* offer, in nats.

    The counterpart's answer is binary: accept or not. The gain is the expected reduction of
    entropy over `param` from observing it:

        EIG = H(belief) − [ P(accept)·H(belief | accept) + P(reject)·H(belief | reject) ]

    An offer the counterpart would accept with probability ~0 or ~1 teaches nothing; the maximum
    is where the answer is genuinely uncertain. That is RQ2's exploration lever: it turns an
    offer into an experiment.
    """
    rv = np.asarray(theta["rv"], dtype=float)
    beta = np.asarray(theta.get("beta", np.ones_like(rv)), dtype=float)
    deadline = np.asarray(theta.get("T", np.full_like(rv, 20.0)), dtype=float)
    w = np.ones_like(rv) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    values = np.asarray(theta[param], dtype=float)

    target = counterpart_target(rv, beta, deadline, round_idx)
    accepts = utility_to_counterpart >= target - 1e-12
    p_accept = float(np.sum(w * accepts))

    h_prior = _entropy(values, w)
    h_accept = _entropy(values, w * accepts) if p_accept > 1e-9 else 0.0
    h_reject = _entropy(values, w * (~accepts)) if p_accept < 1 - 1e-9 else 0.0
    eig = h_prior - (p_accept * h_accept + (1.0 - p_accept) * h_reject)
    return {"eig": float(max(eig, 0.0)), "p_accept": p_accept,
            "entropy_prior": h_prior, "entropy_if_accept": h_accept,
            "entropy_if_reject": h_reject, "param": param}


def evpi(offers: list[Offer], own_utility: Utility, round_idx: int,
         theta: dict[str, np.ndarray], weights: np.ndarray | None = None,
         est_counterpart_utility=None) -> dict:
    """t05 — Expected value of perfect information about θ.

    EVPI = E_θ[ max_o U(o | θ) ] − max_o E_θ[ U(o) ]

    with U(o | θ) = own utility of the offer if the counterpart accepts it under that θ, and
    the own reservation value otherwise. It is how much you would gain if someone told you the
    truth before offering: the upper bound on what *any* probe can be worth.
    """
    rv = np.asarray(theta["rv"], dtype=float)
    beta = np.asarray(theta.get("beta", np.ones_like(rv)), dtype=float)
    deadline = np.asarray(theta.get("T", np.full_like(rv, 20.0)), dtype=float)
    w = np.ones_like(rv) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    target = counterpart_target(rv, beta, deadline, round_idx)

    own_values = np.array([own_utility(o) for o in offers])
    if est_counterpart_utility is None:
        # Without an estimate of the counterpart's utility, the complement is the minimal
        # approximation: what one side gains the other loses (valid in a single-issue
        # distributive case only).
        cp_values = 1.0 - own_values
    else:
        cp_values = np.array([float(est_counterpart_utility(o)) for o in offers])

    fallback = own_utility.reservation_value
    # realised utility per offer and per sample of θ
    accepts = cp_values[:, None] >= target[None, :] - 1e-12
    payoff = np.where(accepts, own_values[:, None], fallback)

    ex_ante = float(np.max(payoff @ w))                     # best offer without knowing θ
    ex_post = float(np.sum(w * payoff.max(axis=0)))         # best offer knowing θ
    best_idx = int(np.argmax(payoff @ w))
    return {"evpi": max(ex_post - ex_ante, 0.0), "best_offer_ex_ante": offers[best_idx],
            "expected_utility_ex_ante": ex_ante, "expected_utility_with_perfect_info": ex_post}
