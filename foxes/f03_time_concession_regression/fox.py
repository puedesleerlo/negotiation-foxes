"""f03 — Time-dependent tactic regression: beta (concession shape), T (deadline) and rv.

Lineage: Faratin, Sierra and Jennings 1998, in the form Baarslag's survey reproduces (§3.6):

    u(t) = rv + (1 - rv) * (1 - [k + (1 - k) * t^(1/e)]),   t = round / T

(rv, e, T) are fitted over the curve of utility that the counterpart's offers give to itself,
and the uncertainty comes from a grid posterior with a Gaussian likelihood whose noise is
calibrated on the synthetic set.

Identifiability warning (survey §5.2 and L1 §5 of the report): rv and T are coupled — the
counterpart concedes towards rv *as* T approaches — so with few rounds many (rv, T) pairs
explain the same curve. The fox reports that correlation in its diagnostics instead of hiding
it behind a point fit. Validated for `beta` only: the deadline `T` does not beat the control.

Fidelity: the functional form is the paper's; the grid posterior and its calibration are this
project's construction. The synthetic generator uses the same family of curves, so the
in-family validation is optimistic (see REVIEW.md, F11).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from foxes.base import BaseFox, BeliefState, Diagnostics, FoxMeta, Posterior, counterpart_offers

ASSETS = Path(__file__).parent.parent.parent / "skills" / "foxes" / \
    "f03_time_concession_regression" / "assets" / "calibration.json"

META = FoxMeta(
    fox_id="f03_time_concession_regression", version="0.2.0",
    estimates=["theta.beta", "theta.T", "theta.rv"], phase="online",
    inputs_required=["estimated utility of each counterpart offer to the counterpart",
                     "round number of each offer"],
    scope_conditions=["a counterpart with a time-dependent tactic (checked with the RMSE of "
                      "the best fit, not assumed)",
                      ">= 4 observed offers",
                      "rv and T are only separable with enough rounds: see the correlation in "
                      "diagnostics"],
    assumptions=["Faratin–Sierra–Jennings functional form u(t)",
                 "parameters constant within the session"],
    paper_ids=["faratin1998nego", "baarslag2015lear"],
    calibration_dataset="data/synthetic/episodes.parquet",
    validation_report="foxes/f03_time_concession_regression/validation_report.md",
    status="implemented",
)

BOUNDS_LO = np.array([0.0, 0.05, 6.0])    # rv, e, T
BOUNDS_HI = np.array([1.0, 5.00, 40.0])


def curve(params: np.ndarray, rounds: np.ndarray, k: float = 0.0) -> np.ndarray:
    rv, e, T = params
    t = np.clip(rounds / max(T, 1e-6), 0.0, 1.0)
    f = k + (1.0 - k) * t ** (1.0 / max(e, 1e-6))
    return rv + (1.0 - rv) * (1.0 - f)


class TimeConcessionRegression(BaseFox):
    meta = META

    def __init__(self, n_bootstrap: int = 200, seed: int = 0, noise: float | None = None,
                 grid_rv: int = 21, grid_e: int = 15, grid_T: int = 16,
                 max_rmse: float = 0.10) -> None:
        self.n_bootstrap = n_bootstrap   # only used by the legacy bootstrap mode
        self.seed = seed
        # Standard deviation of the observation noise on the concession curve. It sets the
        # width of the posterior and is calibrated on the synthetic set.
        self.noise = noise if noise is not None else self._load_noise()
        self.max_rmse = max_rmse
        self.grid = self._build_grid(grid_rv, grid_e, grid_T)

    @staticmethod
    def _load_noise(default: float = 0.06) -> float:
        if ASSETS.exists():
            return float(json.loads(ASSETS.read_text()).get("noise", default))
        return default

    @staticmethod
    def _build_grid(n_rv: int, n_e: int, n_T: int) -> np.ndarray:
        rv = np.linspace(0.0, 0.95, n_rv)
        e = np.geomspace(0.08, 5.0, n_e)      # log-uniform: e lives on a multiplicative scale
        T = np.linspace(8.0, 40.0, n_T)
        mesh = np.meshgrid(rv, e, T, indexing="ij")
        return np.column_stack([m.ravel() for m in mesh])

    def scope_check(self, state: BeliefState) -> tuple[bool, list[str]]:
        offers = counterpart_offers(state)
        warns: list[str] = []
        if len(offers) < 4:
            warns.append("fewer than 4 observed offers: out of scope")
        if any(o.est_utility_to_proposer is None for o in offers):
            warns.append("missing estimated utilities of the counterpart")
        if not warns:
            # The "time-dependent counterpart" condition is checked, not assumed: if the best
            # curve of the family does not fit, the counterpart is not of that type (tit-for-tat,
            # erratic) and the posterior over (rv, e, T) means nothing.
            rmse = self._best_fit_rmse(state)
            if rmse is not None and rmse > self.max_rmse:
                warns.append(f"the time-dependent curve does not fit (RMSE {rmse:.3f} > "
                             f"{self.max_rmse}): the counterpart does not seem to use that tactic")
        return (not warns), warns

    def _best_fit_rmse(self, state: BeliefState) -> float | None:
        offers = counterpart_offers(state)
        utils = np.array([o.est_utility_to_proposer for o in offers
                          if o.est_utility_to_proposer is not None], dtype=float)
        rounds = np.array([o.round for o in offers
                           if o.est_utility_to_proposer is not None], dtype=float)
        if len(utils) < 3:
            return None
        predicted = np.array([curve(theta, rounds) for theta in self.grid])
        sse = np.sum((predicted - utils[None, :]) ** 2, axis=1)
        return float(np.sqrt(sse.min() / len(utils)))

    @staticmethod
    def _fit(rounds: np.ndarray, utils: np.ndarray, start: np.ndarray) -> np.ndarray:
        res = least_squares(lambda p: curve(p, rounds) - utils, start,
                            bounds=(BOUNDS_LO, BOUNDS_HI), max_nfev=200)
        return res.x

    def posterior(self, state: BeliefState) -> Posterior:
        params = ["rv", "beta", "T"]
        offers = counterpart_offers(state)
        rounds = np.array([o.round for o in offers], dtype=float)
        utils = np.array([o.est_utility_to_proposer if o.est_utility_to_proposer is not None
                          else np.nan for o in offers], dtype=float)
        mask = ~np.isnan(utils)
        rounds, utils = rounds[mask], utils[mask]
        ok, warns = self.scope_check(state)
        rng = np.random.default_rng(self.seed)

        if len(utils) < 3:
            samples = np.column_stack([rng.uniform(0, 1, 500), rng.uniform(0.05, 5, 500),
                                       rng.uniform(6, 40, 500)])
            return Posterior(META.fox_id, META.version, params, samples, scope_ok=False,
                             warnings=warns + ["too few observations: flat prior"],
                             program_id=state.program_id, party=state.party, round=state.round)

        # Grid posterior over (rv, e, T) with a Gaussian likelihood on the curve. The residual
        # bootstrap gave far too narrow intervals: it fitted the curve well but did not capture
        # the ridge of (rv, T) combinations that explain it almost equally well. The grid does,
        # which is exactly what the identifiability problem demands.
        grid = self.grid
        predicted = np.array([curve(theta, rounds) for theta in grid])   # (n_grid, n_obs)
        sse = np.sum((predicted - utils[None, :]) ** 2, axis=1)
        log_w = -sse / (2.0 * self.noise ** 2)
        log_w -= log_w.max()
        weights = np.exp(log_w)
        # The log-uniform prior on e is already in the grid; rv and T are uniform.

        post = Posterior(META.fox_id, META.version, params,
                         grid[:, [0, 1, 2]], weights, scope_ok=ok, warnings=warns,
                         program_id=state.program_id, party=state.party, round=state.round)
        eff = post.effective_sample_size()
        if eff > 5:
            draws = post.resample(2000, rng)
            corr = float(np.corrcoef(draws[:, 0], draws[:, 2])[0, 1])
            post.warnings = post.warnings + [
                f"rv-T correlation in the posterior: {corr:.2f} "
                f"(structural coupling: see scope)"]
        return post

    def diagnostics(self, state: BeliefState) -> Diagnostics:
        diag = super().diagnostics(state)
        post = self.posterior(state)
        if len(post.samples) > 5:
            diag.extra["corr_rv_T"] = float(np.corrcoef(post.samples[:, 0], post.samples[:, 2])[0, 1])
        return diag
