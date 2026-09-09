"""f07 — ZOPA, Pareto frontier, Nash point and joint gains by Monte Carlo propagation.

Lineage: the Adversarial Risk Analysis pattern (banks2020adve): represent what is unknown about
the counterpart with subjective distributions and *propagate* them by simulation down to the
quantity the decision depends on, instead of collapsing them into a point.

Input: the declared own utility and samples of the counterpart's θ (weights, directions and rv)
coming from the fox pool. Output: a distribution over the size of the ZOPA, the utility reachable
on the frontier and the probability that no zone of agreement exists.
"""
from __future__ import annotations

import numpy as np

from foxes.base import BaseFox, BeliefState, FoxMeta, Posterior
from foxes.domain import Domain, Offer, Utility, nash_point, pareto_frontier

META = FoxMeta(
    fox_id="f07_zopa_pareto_estimator", version="0.1.0",
    estimates=["zopa_size", "u_own_at_nash", "p_no_zopa", "joint_at_nash"], phase="both",
    inputs_required=["declared own utility", "samples of the counterpart's θ"],
    scope_conditions=["additive utilities", "enumerable or samplable outcome space",
                      "the θ samples must come from in-scope foxes"],
    assumptions=["the counterpart's utility is additive over the same issues",
                 "the preference directions per issue are in the θ samples"],
    paper_ids=["banks2020adve"],
    calibration_dataset="data/synthetic/episodes.parquet",
    validation_report="foxes/f07_zopa_pareto_estimator/validation_report.md",
    status="implemented",
)


def utility_from_theta(domain: Domain, weights: np.ndarray, directions: np.ndarray,
                       rv: float) -> Utility:
    value_maps: dict[str, dict | tuple] = {}
    for k, issue in enumerate(domain.issues):
        direction = float(directions[k]) if k < len(directions) else 1.0
        if issue.type == "continuous":
            value_maps[issue.issue_id] = ((issue.low, issue.high) if direction > 0
                                          else (issue.high, issue.low))
        else:
            scores = np.linspace(0.0, 1.0, len(issue.values))
            if direction < 0:
                scores = scores[::-1]
            value_maps[issue.issue_id] = {v: float(s) for v, s in zip(issue.values, scores)}
    util = Utility(domain, dict(zip(domain.issue_ids, weights)), value_maps)
    util.reservation_value = float(rv)
    return util


class ZopaParetoEstimator(BaseFox):
    meta = META

    def __init__(self, n_draws: int = 200, space_cap: int = 400, seed: int = 0) -> None:
        self.n_draws = n_draws
        self.space_cap = space_cap
        self.seed = seed

    def scope_check(self, state: BeliefState) -> tuple[bool, list[str]]:
        warns: list[str] = []
        if state.own_utility is None:
            warns.append("missing declared own utility")
        if "theta_samples" not in state.data:
            warns.append("missing samples of the counterpart's θ")
        return (not warns), warns

    def posterior(self, state: BeliefState) -> Posterior:
        params = ["zopa_fraction", "u_own_at_nash", "u_other_at_nash", "joint_at_nash"]
        ok, warns = self.scope_check(state)
        if not ok:
            return Posterior(META.fox_id, META.version, params,
                             np.zeros((1, len(params))), scope_ok=False, warnings=warns,
                             program_id=state.program_id, party=state.party, round=state.round)

        rng = np.random.default_rng(self.seed)
        domain, own = state.domain, state.own_utility
        space = domain.outcome_space()
        if len(space) > self.space_cap:
            idx = rng.choice(len(space), self.space_cap, replace=False)
            space = [space[i] for i in idx]
        own_vals = np.array([own(o) for o in space])

        theta = state.data["theta_samples"]      # (n, dim): weights + directions + rv
        rows = []
        n = min(self.n_draws, len(theta["weights"]))
        for i in range(n):
            other = utility_from_theta(domain, theta["weights"][i], theta["directions"][i],
                                       theta["rv"][i])
            other_vals = np.array([other(o) for o in space])
            feasible = (own_vals >= own.reservation_value) & (other_vals >= other.reservation_value)
            zopa_fraction = float(feasible.mean())
            if feasible.any():
                sub = [space[j] for j in np.flatnonzero(feasible)]
                nash_offer, _ = nash_point(sub, own, other)
                u_own, u_other = own(nash_offer), other(nash_offer)
            else:
                u_own, u_other = own.reservation_value, other.reservation_value
            rows.append([zopa_fraction, u_own, u_other, u_own + u_other])
        return Posterior(META.fox_id, META.version, params, np.array(rows), scope_ok=ok,
                         warnings=warns, program_id=state.program_id, party=state.party,
                         round=state.round)

    def p_no_zopa(self, state: BeliefState) -> float:
        post = self.posterior(state)
        return float(np.sum(post.weights * (post.column("zopa_fraction") <= 0.0)))

    def true_frontier(self, domain: Domain, ua: Utility, ub: Utility) -> list[Offer]:
        """Exact frontier, to compare the estimate against the truth in feedback."""
        return pareto_frontier(domain.outcome_space(), ua, ub)
