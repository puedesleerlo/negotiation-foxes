"""f09 — Issue weights by Bayesian updating over a hypothesis space.

Lineage: Konishi et al. 2026 (§3.1) and Hindriks and Tykhonov 2008. Each hypothesis h is a
weight vector plus a preference direction per issue. The likelihood of an observed offer is
Luce's choice rule over the outcome space:

    p(o | h) = exp(u_h(o) / sigma) / sum_{o' in space} exp(u_h(o') / sigma)

sigma is the dispersion: low sigma, a near-optimal counterpart; high sigma, a noisy one. It
plays the same role as Hindriks and Tykhonov's dispersion parameter and is calibrated on the
synthetic set.

A mechanism *independent* of f02: f02 looks at how much is conceded per issue, f09 at how
likely each offer is under each hypothesis. Their disagreement is informative (protocol §6.5,
rule 4). The linguistic likelihood of Konishi's work (which requires an LLM) is not part of
this version.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from foxes.base import BaseFox, BeliefState, FoxMeta, Posterior, counterpart_offers
from foxes.domain import Domain, Utility

ASSETS = Path(__file__).parent.parent.parent / "skills" / "foxes" / \
    "f09_hypothesis_issue_weights" / "assets" / "calibration.json"

META = FoxMeta(
    fox_id="f09_hypothesis_issue_weights", version="0.1.0",
    estimates=["theta.w"], phase="online",
    inputs_required=["the counterpart's offers", "the domain's outcome space"],
    scope_conditions=["multi-issue (>= 2 issues)", ">= 2 observed offers",
                      "additive utility of the counterpart"],
    assumptions=["the counterpart proposes offers of high utility to itself",
                 "the hypothesis space discretises the simplex with enough resolution",
                 "the value functions per issue are monotone in the declared order"],
    paper_ids=["konishi2026pref", "hindriks2008oppo"],
    calibration_dataset="data/synthetic/episodes.parquet",
    validation_report="foxes/f09_hypothesis_issue_weights/validation_report.md",
    status="implemented",
)


def _value_map(domain: Domain, issue_id: str, direction: int) -> dict | tuple:
    issue = next(i for i in domain.issues if i.issue_id == issue_id)
    if issue.type == "continuous":
        return (issue.low, issue.high) if direction > 0 else (issue.high, issue.low)
    values = list(issue.values)
    scores = np.linspace(0.0, 1.0, len(values))
    if direction < 0:
        scores = scores[::-1]
    return {v: float(s) for v, s in zip(values, scores)}


class HypothesisIssueWeights(BaseFox):
    meta = META

    def __init__(self, n_hypotheses: int = 600, sigma: float | None = None, seed: int = 0,
                 space_cap: int = 400) -> None:
        self.n_hypotheses = n_hypotheses
        # sigma is the dispersion of the Luce likelihood: low = near-optimal counterpart,
        # high = noisy. Calibrated on the synthetic set for nominal coverage.
        self.sigma = sigma if sigma is not None else self._load_sigma()
        self.seed = seed
        self.space_cap = space_cap

    @staticmethod
    def _load_sigma(default: float = 0.08) -> float:
        if ASSETS.exists():
            return float(json.loads(ASSETS.read_text()).get("sigma", default))
        return default

    def scope_check(self, state: BeliefState) -> tuple[bool, list[str]]:
        warns: list[str] = []
        if len(state.domain.issues) < 2:
            warns.append("single-issue domain: weights are not defined")
        if len(counterpart_offers(state)) < 2:
            warns.append("fewer than 2 observed offers: out of scope")
        return (not warns), warns

    def _hypotheses(self, domain: Domain) -> tuple[np.ndarray, list[list[int]]]:
        rng = np.random.default_rng(self.seed)
        n_issues = len(domain.issues)
        weights = rng.dirichlet(np.ones(n_issues), self.n_hypotheses)
        directions = [list(rng.choice([-1, 1], size=n_issues)) for _ in range(self.n_hypotheses)]
        return weights, directions

    def posterior(self, state: BeliefState) -> Posterior:
        domain = state.domain
        issue_ids = domain.issue_ids
        params = [f"w.{i}" for i in issue_ids]
        ok, warns = self.scope_check(state)
        offers = [o.offer for o in counterpart_offers(state) if o.offer is not None]
        weights_h, directions = self._hypotheses(domain)

        if not offers:
            return Posterior(META.fox_id, META.version, params, weights_h, scope_ok=False,
                             warnings=warns + ["no observations: Dirichlet(1) prior"],
                             program_id=state.program_id, party=state.party, round=state.round)

        rng = np.random.default_rng(self.seed + 1)
        space = domain.outcome_space()
        if len(space) > self.space_cap:
            idx = rng.choice(len(space), self.space_cap, replace=False)
            space = [space[i] for i in idx]

        log_post = np.zeros(len(weights_h))
        for h_idx in range(len(weights_h)):
            vm = {i: _value_map(domain, i, directions[h_idx][k])
                  for k, i in enumerate(issue_ids)}
            util = Utility(domain, dict(zip(issue_ids, weights_h[h_idx])), vm)
            space_u = np.array([util(o) for o in space])
            log_z = np.log(np.sum(np.exp(space_u / self.sigma)) + 1e-300)
            for offer in offers:
                log_post[h_idx] += util(offer) / self.sigma - log_z
        log_post -= log_post.max()
        return Posterior(META.fox_id, META.version, params, weights_h, np.exp(log_post),
                         scope_ok=ok, warnings=warns, program_id=state.program_id,
                         party=state.party, round=state.round)
