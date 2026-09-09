"""f02 — Issue weights from the normalised concession ratio.

Lineage: Baarslag's survey §5.3 (Jonker et al.; Carbonneau and Vahidov; Niemann and Lang). The
operational idea the survey collects is direct: *one concedes first on the issues that matter
least*, so with the normalised concession ratio c_i between consecutive offers,

    w_i ∝ 1 - c_i,   normalised to sum 1.

It needs no knowledge of the counterpart's value functions: it measures the movement on each
issue's raw scale. Uncertainty comes from a Dirichlet whose concentration is calibrated on the
synthetic dataset — the bootstrap alone measured only sampling noise, not the model error of
w_i = 1 − c_i, and was overconfident (90 % coverage 0.63 before calibration).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from foxes.base import BaseFox, BeliefState, FoxMeta, Posterior, counterpart_offers
from foxes.domain import Domain

ASSETS = Path(__file__).parent.parent.parent / "skills" / "foxes" / \
    "f02_concession_issue_weights" / "assets" / "calibration.json"

META = FoxMeta(
    fox_id="f02_concession_issue_weights", version="0.1.0",
    estimates=["theta.w"], phase="online",
    inputs_required=[">= 3 consecutive offers by the counterpart", "range of each issue"],
    scope_conditions=["multi-issue (>= 2 issues)", ">= 3 observed offers",
                      "additive utility of the counterpart"],
    assumptions=["one concedes first on the least important issues",
                 "changes on each issue's raw scale are comparable across issues"],
    paper_ids=["baarslag2015lear"],
    calibration_dataset="data/synthetic/episodes.parquet",
    validation_report="foxes/f02_concession_issue_weights/validation_report.md",
    status="implemented",
)


def _normalized_change(domain: Domain, issue_id: str, a, b) -> float:
    issue = next(i for i in domain.issues if i.issue_id == issue_id)
    if issue.type == "continuous":
        span = max(issue.high - issue.low, 1e-9)
        return float(abs(float(b) - float(a)) / span)
    values = list(issue.values)
    if len(values) <= 1:
        return 0.0
    return float(abs(values.index(b) - values.index(a)) / (len(values) - 1))


class ConcessionIssueWeights(BaseFox):
    meta = META

    def __init__(self, n_bootstrap: int = 400, seed: int = 0, floor: float = 0.02,
                 concentration: float | None = None) -> None:
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.floor = floor
        # The calibrated concentration turns the point estimate into a distribution with
        # nominal coverage. The bootstrap only measures sampling noise, not the error of the
        # model w_i = 1 - c_i, which is the dominant term: hence calibration on the synthetic set.
        self.concentration = concentration if concentration is not None else self._load_kappa()

    @staticmethod
    def _load_kappa(default: float = 12.0) -> float:
        if ASSETS.exists():
            return float(json.loads(ASSETS.read_text()).get("concentration", default))
        return default

    def scope_check(self, state: BeliefState) -> tuple[bool, list[str]]:
        warns: list[str] = []
        if len(state.domain.issues) < 2:
            warns.append("single-issue domain: weights are not defined")
        if len(counterpart_offers(state)) < 3:
            warns.append("fewer than 3 observed offers: out of scope")
        return (not warns), warns

    def posterior(self, state: BeliefState) -> Posterior:
        domain = state.domain
        issue_ids = domain.issue_ids
        params = [f"w.{i}" for i in issue_ids]
        offers = [o.offer for o in counterpart_offers(state) if o.offer is not None]
        ok, warns = self.scope_check(state)
        rng = np.random.default_rng(self.seed)

        pairs = [(a, b) for a, b in zip(offers, offers[1:])]
        if not pairs:
            samples = rng.dirichlet(np.ones(len(issue_ids)), self.n_bootstrap)
            return Posterior(META.fox_id, META.version, params, samples, scope_ok=False,
                             warnings=warns + ["no offer pairs: flat prior"],
                             program_id=state.program_id, party=state.party, round=state.round)

        changes = np.array([[_normalized_change(domain, i, a[i], b[i]) for i in issue_ids]
                            for a, b in pairs])
        draws = []
        for _ in range(self.n_bootstrap):
            idx = rng.integers(0, len(pairs), len(pairs))
            c = changes[idx].mean(axis=0)
            c = c / max(c.max(), 1e-9)          # concession relative to the most-conceded issue
            w = np.clip(1.0 - c, self.floor, None)
            point = w / w.sum()
            # Dirichlet centred on the point, with the calibrated concentration.
            draws.append(rng.dirichlet(np.clip(point * self.concentration, 0.05, None)))
        return Posterior(META.fox_id, META.version, params, np.array(draws), scope_ok=ok,
                         warnings=warns, program_id=state.program_id, party=state.party,
                         round=state.round)
