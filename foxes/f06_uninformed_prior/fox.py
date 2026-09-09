"""f06 — Uninformed prior (the mandatory control).

Runs always, for every parameter. It is the baseline that makes every other score meaningful:
without it a log score or a CRPS says nothing about how much information a fox added
(protocol §6.5, rule 2).

The draw is a deterministic function of the state (seed, round, number of observations), so
calling `posterior()` twice on the same state returns the same samples: the entropy reported
in a fox_call event is the entropy of the samples that actually entered the pool.
"""
from __future__ import annotations

import numpy as np

from foxes.base import BaseFox, BeliefState, FoxMeta, Posterior

META = FoxMeta(
    fox_id="f06_uninformed_prior", version="0.1.1",
    estimates=["theta.rv", "theta.w", "theta.beta", "theta.T"], phase="both",
    inputs_required=["declared support of each parameter"],
    scope_conditions=["always applicable"],
    assumptions=["none beyond the declared support"],
    paper_ids=[],
    calibration_dataset=None,
    validation_report="foxes/f06_uninformed_prior/validation_report.md",
    status="implemented",
)

DEFAULT_SUPPORT = {"rv": (0.0, 1.0), "beta": (0.05, 5.0), "T": (5.0, 40.0), "w": (0.0, 1.0)}


class UninformedPrior(BaseFox):
    meta = META

    def __init__(self, params: list[str] | None = None, support: dict | None = None,
                 n: int = 2000, seed: int = 0) -> None:
        self.params = params or ["rv"]
        self.support = {**DEFAULT_SUPPORT, **(support or {})}
        self.n = n
        self.seed = seed

    def _rng(self, state: BeliefState) -> np.random.Generator:
        """A generator keyed on the state, so the same state always yields the same draw."""
        key = (self.seed * 1_000_003 + int(state.round) * 1_009 + len(state.observations)) % (1 << 31)
        return np.random.default_rng(key)

    def posterior(self, state: BeliefState) -> Posterior:
        rng = self._rng(state)
        cols = []
        for param in self.params:
            base = param.split(".")[0]
            lo, hi = self.support.get(base, (0.0, 1.0))
            cols.append(rng.uniform(lo, hi, self.n))
        samples = np.column_stack(cols)
        if any(p.startswith("w") for p in self.params) and len(self.params) > 1:
            # Weights live on the simplex: the uninformed prior is Dirichlet(1).
            samples = rng.dirichlet(np.ones(len(self.params)), self.n)
        return Posterior(META.fox_id, META.version, list(self.params), samples,
                         scope_ok=True, program_id=state.program_id,
                         party=state.party, round=state.round)
