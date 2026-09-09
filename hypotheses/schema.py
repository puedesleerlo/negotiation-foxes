"""Schema of the hypothesis platform (§9).

A hypothesis is a *falsifiable* claim about θ, about the counterpart's behaviour or about the
effectiveness of a strategy, with an associated test and a resolution in feedback. Without a
test it is not a hypothesis: it is an opinion, and it does not enter.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

HypothesisType = Literal["parameter", "behavior", "strategy"]
HypothesisStatus = Literal["open", "supported", "refuted", "unresolved"]
TestKind = Literal["probe_offer", "observe_rounds", "meso", "contingent_clause"]


class HypothesisTest(BaseModel):
    """How the hypothesis is discriminated. `resolves_if` says which observation moves the belief."""

    kind: TestKind
    spec: dict = Field(default_factory=dict)
    resolves_if: str
    p_true_if_confirmed: float = 0.9
    p_true_if_disconfirmed: float = 0.1

    @field_validator("p_true_if_confirmed", "p_true_if_disconfirmed")
    @classmethod
    def _in_unit(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("probabilities must lie in [0, 1]")
        return v


class Threshold(BaseModel):
    """The bound that makes a hypothesis about a numeric parameter resolvable.

    Without it a hypothesis stays prose and the feedback cannot score it, which is the same as
    not having registered it (§9).
    """

    comparison: Literal["gte", "lte", "between"]
    values: list[float]
    unit: str = "USD"

    def holds(self, truth: float) -> bool:
        if self.comparison == "gte":
            return truth >= self.values[0]
        if self.comparison == "lte":
            return truth <= self.values[0]
        low, high = min(self.values[:2]), max(self.values[:2])
        return low <= truth <= high

    def describe(self) -> str:
        if self.comparison == "between":
            return f"between {min(self.values[:2]):,.0f} and {max(self.values[:2]):,.0f} {self.unit}"
        sign = "≥" if self.comparison == "gte" else "≤"
        return f"{sign} {self.values[0]:,.0f} {self.unit}"


class Resolution(BaseModel):
    at: str                       # "round:9" | "feedback"
    evidence: str
    truth_value: Optional[bool] = None
    note: Optional[str] = None


class Hypothesis(BaseModel):
    hypothesis_id: str
    program_id: str
    party: str
    created_at: str               # "prep" | "round:<k>"
    type: HypothesisType
    statement: str
    variable: Optional[str] = None       # e.g. "theta.rv"
    interval: Optional[tuple[float, float]] = None
    threshold: Optional[Threshold] = None
    prior_belief: float = 0.5            # p(the claim is true)
    current_belief: float = 0.5
    evidence: list[str] = Field(default_factory=list)   # "fox:f01@0.1.0", "corpus:..."
    test: HypothesisTest
    status: HypothesisStatus = "open"
    updates: list[dict] = Field(default_factory=list)
    resolution: Optional[Resolution] = None
    brier: Optional[float] = None

    @field_validator("prior_belief", "current_belief")
    @classmethod
    def _in_unit(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("beliefs must lie in [0, 1]")
        return v

    def is_resolved(self) -> bool:
        return self.status in ("supported", "refuted")
