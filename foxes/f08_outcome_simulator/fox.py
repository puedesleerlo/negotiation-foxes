"""f08 — Outcome simulator per candidate strategy.

Lineage: the experimental design of hendrikx2012eval (fix everything except the counterpart
family and the scenario) applied forwards: given an own candidate strategy and samples of θ,
whole negotiations are simulated against counterpart families and the distribution of own
utility, joint utility and impasse probability is returned.

Status `implemented`, not `validated`: its output is conditional on the assumption that the
real counterpart belongs to one of the simulated families. Validating it honestly requires
contrasting it against episodes whose counterpart belongs to none, and that belongs to Part 3,
with complete programs.
"""
from __future__ import annotations

import numpy as np

from foxes.base import BaseFox, BeliefState, FoxMeta, Posterior
from foxes.domain import Counterpart, Domain, Utility, target_utility
from foxes.f07_zopa_pareto_estimator.fox import utility_from_theta

META = FoxMeta(
    fox_id="f08_outcome_simulator", version="0.1.0",
    estimates=["u_own", "joint_utility", "p_impasse", "rounds"], phase="prep",
    inputs_required=["own candidate strategy (e, deadline)", "samples of θ",
                     "counterpart families to consider"],
    scope_conditions=["alternating-offers protocol",
                      "the real counterpart belongs to one of the simulated families"],
    assumptions=["the simulated families cover the real behaviour",
                 "the declared own utility is correct"],
    paper_ids=["hendrikx2012eval", "faratin1998nego"],
    calibration_dataset="data/synthetic/episodes.parquet",
    validation_report="foxes/f08_outcome_simulator/validation_report.md",
    status="implemented",
)


class OutcomeSimulator(BaseFox):
    meta = META

    def __init__(self, n_draws: int = 100, max_rounds: int = 24, space_cap: int = 300,
                 seed: int = 0) -> None:
        self.n_draws = n_draws
        self.max_rounds = max_rounds
        self.space_cap = space_cap
        self.seed = seed

    def scope_check(self, state: BeliefState) -> tuple[bool, list[str]]:
        warns: list[str] = []
        if state.own_utility is None:
            warns.append("missing declared own utility")
        if "theta_samples" not in state.data:
            warns.append("missing samples of the counterpart's θ")
        if "strategy" not in state.data:
            warns.append("missing candidate strategy (e, deadline)")
        return (not warns), warns

    def simulate(self, domain: Domain, own: Utility, other: Utility, family: str,
                 own_e: float, own_deadline: int, rng: np.random.Generator) -> dict:
        space = domain.outcome_space()
        if len(space) > self.space_cap:
            idx = rng.choice(len(space), self.space_cap, replace=False)
            space = [space[i] for i in idx]
        own_vals = np.array([own(o) for o in space])
        other_vals = np.array([other(o) for o in space])
        cp = Counterpart(utility=other, family=family, e=1.0 if family == "tit_for_tat" else 0.5,
                         deadline=int(rng.integers(12, 31)),
                         rng=np.random.default_rng(int(rng.integers(1 << 31))))
        concession = 0.0
        last_u_to_own = None
        for round_idx in range(1, self.max_rounds + 1):
            t = min(round_idx / own_deadline, 1.0)
            target = target_utility(t, own_e, p_min=own.reservation_value, p_max=1.0)
            offer = space[int(np.argmin(np.abs(own_vals - target)))]
            if cp.accepts(offer, round_idx, concession):
                return {"agreement": 1.0, "u_own": float(own(offer)),
                        "u_other": float(other(offer)), "rounds": float(round_idx)}
            cp_target = cp.target(round_idx, concession)
            cp_offer = cp.choose_offer(space, cp_target, other_vals)
            u_to_own = float(own(cp_offer))
            if last_u_to_own is not None:
                concession = max(u_to_own - last_u_to_own, 0.0)
            last_u_to_own = u_to_own
            if u_to_own >= max(own.reservation_value, target - 1e-9):
                return {"agreement": 1.0, "u_own": u_to_own,
                        "u_other": float(other(cp_offer)), "rounds": float(round_idx)}
        return {"agreement": 0.0, "u_own": own.reservation_value,
                "u_other": other.reservation_value, "rounds": float(self.max_rounds)}

    def posterior(self, state: BeliefState) -> Posterior:
        params = ["u_own", "u_other", "joint_utility", "agreement", "rounds"]
        ok, warns = self.scope_check(state)
        if not ok:
            return Posterior(META.fox_id, META.version, params, np.zeros((1, len(params))),
                             scope_ok=False, warnings=warns, program_id=state.program_id,
                             party=state.party, round=state.round)
        rng = np.random.default_rng(self.seed)
        theta = state.data["theta_samples"]
        strategy = state.data["strategy"]
        families = state.data.get("families", ["boulware", "conceder", "linear", "tit_for_tat"])
        rows = []
        n = min(self.n_draws, len(theta["weights"]))
        for i in range(n):
            other = utility_from_theta(state.domain, theta["weights"][i],
                                       theta["directions"][i], theta["rv"][i])
            family = families[i % len(families)]
            res = self.simulate(state.domain, state.own_utility, other, family,
                                float(strategy["e"]), int(strategy["deadline"]), rng)
            rows.append([res["u_own"], res["u_other"], res["u_own"] + res["u_other"],
                         res["agreement"], res["rounds"]])
        return Posterior(META.fox_id, META.version, params, np.array(rows), scope_ok=ok,
                         warnings=warns, program_id=state.program_id, party=state.party,
                         round=state.round)

    def p_impasse(self, state: BeliefState) -> float:
        post = self.posterior(state)
        return float(np.sum(post.weights * (post.column("agreement") < 0.5)))
