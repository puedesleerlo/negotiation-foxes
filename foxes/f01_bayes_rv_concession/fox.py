"""f01 — The counterpart's reservation value by Bayesian updating on its concession.

Lineage: Zeng and Sycara 1998 (Bayesian learning over a hypothesis space of the RV), as
formulated in Baarslag's survey (§4.1 and §5.1.1): the counterpart never offers below its
reservation value and concedes towards it as its deadline approaches.

Fidelity: *lineage*, not reproduction. The sufficient statistic and the likelihood below are
this project's construction, inspired by those sources — not the discrete hypothesis network of
the original paper (see REVIEW.md, F12).

Sufficient statistic: m_t = the lowest utility (to itself) the counterpart has offered itself up
to round t. The likelihood is

    p(m_t | rv, t) ∝ 1[rv <= m_t] · Exp(m_t - rv ; scale = g0·(1 - t))

i.e. rv lies below the lowest offer observed, and the gap between the two shrinks as t -> 1.
The private deadline that defines t is unknown and coupled with rv (survey §5.2), so it is
marginalised over a grid of plausible deadlines. Recomputed from the full history on every
call (idempotent and replay-safe), not by accumulating updates.
"""
from __future__ import annotations

import numpy as np

from foxes.base import BaseFox, BeliefState, FoxMeta, Posterior, counterpart_offers

META = FoxMeta(
    fox_id="f01_bayes_rv_concession", version="0.1.0",
    estimates=["theta.rv"], phase="online",
    inputs_required=["the counterpart's offers with their estimated utility to it",
                     "the protocol's maximum number of rounds"],
    scope_conditions=["a counterpart that concedes with a trend (first third of its offers at "
                      "least 0.03 above the last third, in its utility)",
                      ">= 2 observed offers",
                      "rv stationary within the session"],
    assumptions=["the counterpart never offers below its rv",
                 "the gap between the lowest offer and the rv shrinks with time",
                 "the estimated utility of the counterpart is approximately right"],
    paper_ids=["zeng1998baye", "baarslag2015lear"],
    calibration_dataset="data/synthetic/episodes.parquet",
    validation_report="foxes/f01_bayes_rv_concession/validation_report.md",
    status="implemented",
)


def trend_concession(utils: list[float]) -> float:
    """Concession *with a trend*: mean of the first third minus mean of the last third.

    The plain range (max - min) confuses noise with concession: a hardliner picking among
    near-equivalent offers produces range without conceding anything. The difference between
    the ends of the series is large only if the counterpart really moved, and in the right
    direction.
    """
    arr = np.asarray(utils, dtype=float)
    if len(arr) < 2:
        return 0.0
    third = max(len(arr) // 3, 1)
    return float(arr[:third].mean() - arr[-third:].mean())


class BayesRvConcession(BaseFox):
    meta = META

    def __init__(self, grid_size: int = 201, gap_scale: float = 0.45,
                 min_scale: float = 0.05, min_concession: float = 0.03,
                 deadline_grid: tuple[int, ...] = tuple(range(10, 41, 2))) -> None:
        self.grid = np.linspace(0.0, 1.0, grid_size)
        self.gap_scale = gap_scale
        self.min_scale = min_scale
        self.min_concession = min_concession
        # The counterpart's private deadline is unknown and coupled with its rv (survey §5.2).
        # Instead of using the protocol horizon as if it were its deadline, marginalise over a
        # grid of possible deadlines.
        self.deadline_grid = np.array(deadline_grid, dtype=float)

    def scope_check(self, state: BeliefState) -> tuple[bool, list[str]]:
        offers = counterpart_offers(state)
        warns: list[str] = []
        if len(offers) < 2:
            warns.append("fewer than 2 observed offers: out of scope")
        utils = [o.est_utility_to_proposer for o in offers]
        if any(u is None for u in utils):
            warns.append("missing estimated utilities of the counterpart")
        elif len(utils) >= 2:
            span = trend_concession(utils)
            if span < self.min_concession:
                # Without concession there is no signal about the rv beyond the bound rv <= m.
                # That is the hardliner case: the fox declares itself out of scope instead of
                # returning a narrow posterior around a bound that carries no information.
                warns.append(f"trend concession {span:.3f} < {self.min_concession}: no "
                             "concession signal, out of scope")
            increases = sum(1 for a, b in zip(utils, utils[1:]) if b > a + 1e-6)
            if len(utils) >= 3 and increases > len(utils) * 0.4:
                warns.append("the counterpart does not concede monotonically: scope doubtful")
        return (not warns), warns

    def posterior(self, state: BeliefState) -> Posterior:
        offers = counterpart_offers(state)
        utils = np.array([o.est_utility_to_proposer for o in offers
                          if o.est_utility_to_proposer is not None], dtype=float)
        ok, warns = self.scope_check(state)
        prior = np.ones_like(self.grid)
        if len(utils) == 0:
            return Posterior(META.fox_id, META.version, ["rv"], self.grid.reshape(-1, 1),
                             prior, scope_ok=False,
                             warnings=["no observations: the flat prior is returned"],
                             program_id=state.program_id, party=state.party, round=state.round)

        m = float(np.min(utils))
        span = trend_concession(list(utils))
        max_rounds = next((o.max_rounds for o in reversed(offers) if o.max_rounds), None)
        t = min(state.round / max_rounds, 1.0) if max_rounds else min(len(utils) / 20.0, 1.0)

        if span < self.min_concession:
            # Only the hard bound rv <= m applies; the rest stays flat. It is all the
            # observation supports when the counterpart has not conceded anything.
            weights = np.where(self.grid <= m + 1e-9, 1.0, 0.0)
            if weights.sum() <= 0:
                weights = prior
            return Posterior(META.fox_id, META.version, ["rv"], self.grid.reshape(-1, 1),
                             weights, scope_ok=False, warnings=warns,
                             program_id=state.program_id, party=state.party, round=state.round)

        gap = m - self.grid
        # p(m | rv) = sum_T p(T) * Exp(m - rv ; scale = g0 * (1 - round/T))
        like = np.zeros_like(self.grid)
        deadlines = self.deadline_grid[self.deadline_grid >= max(state.round, 1)]
        if deadlines.size == 0:
            deadlines = self.deadline_grid[-1:]
        for deadline in deadlines:
            progress = min(state.round / deadline, 1.0)
            scale_T = max(self.gap_scale * (1.0 - progress), self.min_scale)
            like += np.where(gap >= 0.0, np.exp(-gap / scale_T) / scale_T, 0.0)
        like /= len(deadlines)
        scale = max(self.gap_scale * (1.0 - t), self.min_scale)
        # A counterpart that conceded and then stopped is close to its rv: the scale narrows,
        # but only if there really was concession before (otherwise it is a hardliner).
        if len(utils) >= 3 and span >= 0.10 and float(np.ptp(utils[-3:])) < 0.02:
            like *= np.exp(-np.clip(gap, 0, None) / max(scale * 0.5, self.min_scale))
        weights = prior * like
        if weights.sum() <= 0:
            weights = prior
            warns = warns + ["degenerate likelihood: the prior is returned"]
        return Posterior(META.fox_id, META.version, ["rv"], self.grid.reshape(-1, 1), weights,
                         scope_ok=ok, warnings=warns, program_id=state.program_id,
                         party=state.party, round=state.round)
