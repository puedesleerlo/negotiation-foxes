"""Personalities: the parametrisation of the agent that decides (§7.2).

A personality is NOT a prompt: it is the rule that scores candidate offers, with its parameters
exposed for sweeping. The LLM writes and argues; the choice of action comes from here, auditable
and reproducible.

`econ`'s rule:

    score(o) = U_own(o) · P(accept | o, rv at its favourable quantile) + λ · EIG(o)

ties broken in favour of the higher expected joint utility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from foxes.domain import Offer, Utility
from tools.decision import expected_information_gain, p_accept_from_belief

PERSONALITIES_DIR = Path("agents/personalities")


@dataclass
class Candidate:
    offer: Offer
    u_own: float
    p_accept: float
    p_accept_optimistic: float
    eig: float
    score: float
    expected_joint: float
    detail: dict = field(default_factory=dict)


@dataclass
class Personality:
    personality_id: str
    spec: dict

    @classmethod
    def load(cls, personality_id: str = "econ",
             directory: Path = PERSONALITIES_DIR) -> "Personality":
        spec = yaml.safe_load((directory / f"{personality_id}.yaml").read_text(encoding="utf-8"))
        return cls(personality_id=spec["personality_id"], spec=spec)

    # ------------------------------------------------------------------ parameters
    @property
    def tau(self) -> float:
        return float(self.spec.get("optimism_quantile_tau", 0.5))

    @property
    def lambda_info(self) -> float:
        return float(self.spec.get("info_gain_weight_lambda", 0.0))

    @property
    def walk_away_margin(self) -> float:
        return float(self.spec.get("walk_away_margin", 0.0))

    @property
    def disagreement_threshold(self) -> float:
        return float(self.spec.get("disagreement_threshold_sd", 1.0))

    def with_overrides(self, **kwargs: Any) -> "Personality":
        """For sweeping tau and lambda without touching the file."""
        spec = dict(self.spec)
        spec.update(kwargs)
        return Personality(self.personality_id, spec)

    # ------------------------------------------------------------------ optimistic belief
    def optimistic_theta(self, theta: dict[str, np.ndarray],
                         weights: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Replace rv by its favourable quantile (1 − tau), keeping the rest of the belief.

        DIRECTION, which is easy to invert: `rv` is the *counterpart's* reservation utility. A
        low rv means they accept offers that are worse for them, i.e. better for us. That is
        why "optimism tau = 0.75" is implemented as the 0.25 quantile of rv. With tau = 0.5
        there is no optimism and the belief is left untouched.
        """
        if self.tau <= 0.5:
            return theta
        rv = np.asarray(theta["rv"], dtype=float)
        w = np.ones_like(rv) if weights is None else np.asarray(weights, dtype=float)
        w = w / w.sum()
        order = np.argsort(rv)
        cw = np.cumsum(w[order])
        favourable = float(np.interp(1.0 - self.tau, cw, rv[order]))
        out = dict(theta)
        out["rv"] = np.full_like(rv, favourable)
        return out

    # ------------------------------------------------------------------ scoring
    def score_offers(self, offers: list[Offer], own_utility: Utility, round_idx: int,
                     theta: dict[str, np.ndarray], weights: np.ndarray | None = None,
                     est_counterpart_utility: Callable[[Offer], float] | None = None,
                     ) -> list[Candidate]:
        """Score every candidate. `est_counterpart_utility` is how this party estimates what an
        offer is worth to the other side; the `1 - u_own` fallback is only right for a
        single-issue distributive case and exists for the synthetic tests — the negotiator always
        passes its own model."""
        optimistic = self.optimistic_theta(theta, weights)
        out: list[Candidate] = []
        for offer in offers:
            u_own = float(own_utility(offer))
            u_cp = (float(est_counterpart_utility(offer)) if est_counterpart_utility
                    else 1.0 - u_own)
            honest = p_accept_from_belief(u_cp, round_idx, theta, weights)
            optimistic_p = p_accept_from_belief(u_cp, round_idx, optimistic, weights)
            info = expected_information_gain(u_cp, round_idx, theta, weights)
            score = u_own * optimistic_p.value + self.lambda_info * info["eig"]
            out.append(Candidate(
                offer=offer, u_own=u_own, p_accept=honest.value,
                p_accept_optimistic=optimistic_p.value, eig=float(info["eig"]),
                score=float(score), expected_joint=(u_own + u_cp) * honest.value,
                detail={"p_accept_interval": (honest.low, honest.high),
                        "u_counterpart_est": u_cp,
                        "entropy_prior": info["entropy_prior"]}))
        # Ties go to the higher expected joint utility (Pareto efficiency, §7.2).
        out.sort(key=lambda c: (round(c.score, 6), round(c.expected_joint, 6)), reverse=True)
        return out

    # ------------------------------------------------------------------ decisions
    def expected_value_of_continuing(self, candidates: list[Candidate],
                                     own_utility: Utility) -> float:
        """The value of keeping on negotiating.

        `candidates` must be evaluated in the *future* rounds that remain, not the current one:
        the point of waiting is that the counterpart still has to concede. Judging with the
        current round's snapshot makes the agent accept anything above its reserve.

        Uses the probability under the optimistic quantile, not the honest one: the
        personality's optimism is not only a way to pick an offer, it is a way to value waiting.
        With tau = 0.5 both coincide and the rule is neutral again.
        """
        if not candidates:
            return own_utility.reservation_value
        fallback = own_utility.reservation_value
        return max(c.p_accept_optimistic * c.u_own + (1.0 - c.p_accept_optimistic) * fallback
                   for c in candidates)

    def should_accept(self, offer_on_table_utility: float, candidates: list[Candidate],
                      own_utility: Utility) -> tuple[bool, dict]:
        """Accept if what is on the table beats the expected value of continuing."""
        ev_continue = self.expected_value_of_continuing(candidates, own_utility)
        accept = (offer_on_table_utility >= own_utility.reservation_value
                  and offer_on_table_utility >= ev_continue)
        return accept, {"u_offer": offer_on_table_utility, "ev_continue": ev_continue,
                        "rv": own_utility.reservation_value}

    def should_walk_away(self, candidates: list[Candidate], own_utility: Utility,
                         p_no_zopa: float | None = None) -> tuple[bool, dict]:
        """Walk away if nothing reachable **over the whole horizon** beats the reserve plus margin.

        The candidates must come evaluated at the end of the horizon, not in the current round:
        in round 1 the counterpart asks for almost everything whatever its reservation value,
        and judging with that snapshot leaves every negotiation on the first turn.
        """
        best_reachable = max((c.u_own for c in candidates if c.p_accept > 0.05), default=0.0)
        threshold = own_utility.reservation_value + self.walk_away_margin
        walk = best_reachable < threshold
        detail = {"best_reachable": best_reachable, "threshold": threshold}
        if p_no_zopa is not None:
            detail["p_no_zopa"] = p_no_zopa
        return walk, detail

    def concession_target(self, round_idx: int, max_rounds: int, aspiration: float,
                          reservation: float) -> float:
        """This round's target utility according to the planned concession blocks."""
        fraction = round_idx / max(max_rounds, 1)
        concede_to = 1.0
        for block in self.spec.get("concession_blocks", []):
            if fraction <= float(block["until_fraction_of_deadline"]):
                concede_to = float(block["concede_to"])
                break
        return aspiration - concede_to * (aspiration - reservation)
