"""Fox registry: instantiates by id, enforces the catalog status and records every call.

Rules the registry enforces (§6.5 of the protocol):
  * a `candidate` fox cannot be instantiated for use in a program;
  * an `implemented` fox only runs flagged as experimental and does not enter the pool;
  * every call produces a `fox_call` event with inputs, outputs, diagnostics and cost.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from foxes.base import BeliefState, Posterior

CATALOG = Path("catalog/foxes.yaml")


@dataclass
class FoxCall:
    """One call to a fox. It is what the gym's trace persists."""

    fox_id: str
    version: str
    program_id: str | None
    party: str | None
    step: str | None
    round: int | None
    n_observations: int
    scope_ok: bool
    warnings: list[str]
    summary: dict
    seconds: float
    experimental: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str, ensure_ascii=False)


def _builders() -> dict[str, Callable[..., Any]]:
    from foxes.f01_bayes_rv_concession import BayesRvConcession
    from foxes.f02_concession_issue_weights import ConcessionIssueWeights
    from foxes.f03_time_concession_regression import TimeConcessionRegression
    from foxes.f04_accept_boundary_kde import AcceptBoundaryKde
    from foxes.f06_uninformed_prior import UninformedPrior
    from foxes.f07_zopa_pareto_estimator import ZopaParetoEstimator
    from foxes.f08_outcome_simulator import OutcomeSimulator
    from foxes.f09_hypothesis_issue_weights import HypothesisIssueWeights
    return {
        "f01_bayes_rv_concession": BayesRvConcession,
        "f02_concession_issue_weights": ConcessionIssueWeights,
        "f03_time_concession_regression": TimeConcessionRegression,
        "f04_accept_boundary_kde": AcceptBoundaryKde,
        "f06_uninformed_prior": UninformedPrior,
        "f07_zopa_pareto_estimator": ZopaParetoEstimator,
        "f08_outcome_simulator": OutcomeSimulator,
        "f09_hypothesis_issue_weights": HypothesisIssueWeights,
    }


class FoxRegistry:
    def __init__(self, catalog_path: Path = CATALOG) -> None:
        self.catalog = {f["fox_id"]: f for f in
                        (yaml.safe_load(catalog_path.read_text()) or {}).get("foxes", [])}
        self.builders = _builders()
        self.calls: list[FoxCall] = []

    def entry(self, fox_id: str) -> dict:
        if fox_id not in self.catalog:
            raise KeyError(f"{fox_id} is not in the catalog")
        return self.catalog[fox_id]

    def status(self, fox_id: str) -> str:
        return self.entry(fox_id).get("status", "candidate")

    def available(self, phase: str | None = None, min_status: str = "validated") -> list[str]:
        order = {"candidate": 0, "implemented": 1, "validated": 2}
        out = []
        for fox_id, entry in self.catalog.items():
            if order.get(entry.get("status", "candidate"), 0) < order[min_status]:
                continue
            if phase and entry.get("phase") not in (phase, "both"):
                continue
            out.append(fox_id)
        return sorted(out)

    def create(self, fox_id: str, allow_experimental: bool = False, **kwargs: Any):
        status = self.status(fox_id)
        if status == "candidate":
            raise ValueError(
                f"{fox_id} is declared `candidate`: it is not implemented and is not used. "
                f"Reason: {self.entry(fox_id).get('blocked_by', 'no implementation')}")
        if status == "implemented" and not allow_experimental:
            raise ValueError(
                f"{fox_id} is `implemented` but not `validated`: it can only run with "
                f"allow_experimental=True and its output does not enter the pool")
        if fox_id not in self.builders:
            raise KeyError(f"{fox_id} has no registered implementation")
        return self.builders[fox_id](**kwargs)

    def call(self, fox, state: BeliefState, step: str | None = None,
             experimental: bool = False) -> Posterior:
        """Run `posterior` and record the fox_call event."""
        started = time.perf_counter()
        post = fox.posterior(state)
        elapsed = time.perf_counter() - started
        self.calls.append(FoxCall(
            fox_id=fox.meta.fox_id, version=fox.meta.version, program_id=state.program_id,
            party=state.party, step=step, round=state.round,
            n_observations=len(state.observations), scope_ok=post.scope_ok,
            warnings=list(post.warnings), summary=post.summary(), seconds=elapsed,
            experimental=experimental,
        ))
        return post

    def scope_filter(self, posteriors: list[Posterior]) -> tuple[list[Posterior], list[Posterior]]:
        """Split what may enter the pool from what is marked out_of_scope."""
        keep = [p for p in posteriors if p.scope_ok]
        drop = [p for p in posteriors if not p.scope_ok]
        return keep, drop

    def dump_calls(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for call in self.calls:
                fh.write(call.to_json() + "\n")
