"""f04 — The counterpart's acceptance boundary (probability of accepting an offer).

Lineage: Baarslag's survey §4.3 (kernel density estimation) and §5.1.2 (learning the
acceptance strategy by recording which offers were accepted or rejected).

Model: p(accept | offer) = sigmoid( a · (u_cp(offer) - b) + c · t ), with (a, b, c) on a grid
with a prior; the posterior is the grid weights, so the returned probability always comes with
an interval. KDE smoothing enters as a minimum width of the boundary: with few observations
the slope `a` cannot be arbitrarily steep. The population prior over the grid is calibrated on
the synthetic set.

Status `implemented`, not validated: within one session of this protocol the boundary is not
identified (the first responses are almost always rejections), and out of scope it does not
beat predicting the base rate (DECISIONS D-020). The gym feeds it real accept/reject responses
so a future validation has data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from foxes.base import (BaseFox, BeliefState, FoxMeta, Observation, Posterior, ProbEstimate)
from foxes.domain import Offer

ASSETS = Path(__file__).parent.parent.parent / "skills" / "foxes" / \
    "f04_accept_boundary_kde" / "assets" / "prior.json"

META = FoxMeta(
    fox_id="f04_accept_boundary_kde", version="0.1.0",
    estimates=["p_accept"], phase="online",
    inputs_required=[">= 4 observed responses (proposed offer -> accepted/rejected)",
                     "estimated utility of each offer to the counterpart"],
    scope_conditions=[">= 4 observed responses",
                      "the protocol reveals explicit rejection"],
    assumptions=["p(accept) is monotonically increasing in the utility the offer gives the counterpart",
                 "the boundary only moves through time pressure"],
    paper_ids=["baarslag2015lear"],
    calibration_dataset="data/synthetic/moves.parquet",
    validation_report="foxes/f04_accept_boundary_kde/validation_report.md",
    status="implemented",
)


class AcceptBoundaryKde(BaseFox):
    meta = META

    def __init__(self, slope_grid=(4.0, 8.0, 16.0, 32.0), threshold_steps: int = 41,
                 time_grid=(-1.0, -0.5, 0.0, 0.5, 1.0), prior: np.ndarray | None = None) -> None:
        self.slopes = np.array(slope_grid, dtype=float)
        self.thresholds = np.linspace(0.0, 1.0, threshold_steps)
        self.time_coefs = np.array(time_grid, dtype=float)
        # Population prior over (slope, threshold, time coefficient), fitted on the calibration
        # split: with 4 responses per session — almost all rejections — a flat grid does not
        # identify the boundary and overestimates the acceptance probability.
        self.prior = prior if prior is not None else self._load_prior()

    def _load_prior(self) -> np.ndarray | None:
        if ASSETS.exists():
            data = json.loads(ASSETS.read_text())
            arr = np.array(data["log_prior"], dtype=float)
            if len(arr) == len(self._grid()):
                return arr
        return None

    def scope_check(self, state: BeliefState) -> tuple[bool, list[str]]:
        responses = state.data.get("responses", [])
        warns: list[str] = []
        if len(responses) < 4:
            warns.append("fewer than 4 observed responses: out of scope")
        if responses and all(r["accepted"] == responses[0]["accepted"] for r in responses):
            warns.append("all responses are identical: the boundary is not identified")
        return (not warns), warns

    def observe_response(self, state: BeliefState, utility_to_counterpart: float,
                         accepted: bool, t: float) -> BeliefState:
        state.data.setdefault("responses", []).append(
            {"u": float(utility_to_counterpart), "accepted": bool(accepted), "t": float(t)})
        return state

    def update(self, state: BeliefState, obs: Observation) -> BeliefState:
        state = super().update(state, obs)
        if obs.action in ("accept", "reject") and obs.est_utility_to_proposer is not None:
            t = (obs.round / obs.max_rounds) if obs.max_rounds else 0.5
            self.observe_response(state, obs.est_utility_to_proposer,
                                  obs.action == "accept", t)
        return state

    def _grid(self) -> np.ndarray:
        return np.array([[a, b, c] for a in self.slopes for b in self.thresholds
                         for c in self.time_coefs], dtype=float)

    def posterior(self, state: BeliefState) -> Posterior:
        grid = self._grid()
        responses = state.data.get("responses", [])
        ok, warns = self.scope_check(state)
        log_w = np.zeros(len(grid)) if self.prior is None else self.prior.copy()
        for r in responses:
            z = grid[:, 0] * (r["u"] - grid[:, 1]) + grid[:, 2] * r["t"]
            p = 1.0 / (1.0 + np.exp(-z))
            p = np.clip(p, 1e-6, 1 - 1e-6)
            log_w += np.log(p) if r["accepted"] else np.log(1.0 - p)
        log_w -= log_w.max()
        return Posterior(META.fox_id, META.version, ["slope", "threshold", "time_coef"],
                         grid, np.exp(log_w), scope_ok=ok, warnings=warns,
                         program_id=state.program_id, party=state.party, round=state.round)

    def p_accept(self, state: BeliefState, offer: Offer,
                 utility_to_counterpart: float | None = None, t: float = 0.5) -> ProbEstimate:
        if utility_to_counterpart is None:
            raise ValueError("f04 needs the estimated utility of the offer to the counterpart")
        post = self.posterior(state)
        z = post.samples[:, 0] * (utility_to_counterpart - post.samples[:, 1]) + post.samples[:, 2] * t
        p = 1.0 / (1.0 + np.exp(-z))
        order = np.argsort(p)
        cw = np.cumsum(post.weights[order])
        return ProbEstimate(
            value=float(np.sum(post.weights * p)),
            low=float(np.interp(0.05, cw, p[order])),
            high=float(np.interp(0.95, cw, p[order])),
            fox_id=META.fox_id, version=META.version,
            scope_ok=post.scope_ok, warnings=list(post.warnings),
        )
