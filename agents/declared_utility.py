"""A party's declared utility, derived **only from its own corpus** (task 2.4).

The gym seals whatever comes out of here at the end of preparation (§5.0). This version is
deterministic and model-free: it extracts the reservation value from the sentences where the
party's own material states it. The planner produces the same thing with more generality; the
output format does not change, and the preparation memo contrasts the two readings.

Isolation: the function receives a `PartyView`, which by construction contains nothing of the
counterpart. It could not leak even if it tried.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from foxes.domain import Utility
from gym.case import PartyView

MONEY = re.compile(r"\$\s?([0-9][0-9,\.]{2,})")
LIMIT_HINTS = (
    ("absolute minimum", "min"), ("absolute upper limit", "max"), ("firm maximum", "max"),
    ("maximum", "max"), ("minimum", "min"), ("at the most", "max"), ("upper limit", "max"),
)


def _to_float(raw: str) -> float:
    return float(raw.replace(",", "").rstrip("."))


@dataclass
class DeclaredUtility:
    """What the party claims the space of agreements is worth to it."""

    role_id: str
    issue_id: str
    direction: str                    # increasing | decreasing
    reservation_value_raw: float      # in the issue's units (e.g. USD)
    reservation_value_utility: float
    aspiration_utility: float
    range: tuple[float, float]
    provenance: list[str] = field(default_factory=list)
    batna_note: Optional[str] = None

    def to_utility(self, view: PartyView) -> Utility:
        low, high = self.range
        value_map = (low, high) if self.direction == "increasing" else (high, low)
        util = Utility(view.domain, {self.issue_id: 1.0}, {self.issue_id: value_map})
        util.reservation_value = self.reservation_value_utility
        return util

    def to_dict(self) -> dict:
        return {
            "role_id": self.role_id, "issue_id": self.issue_id, "direction": self.direction,
            "reservation_value_raw": self.reservation_value_raw,
            "reservation_value_utility": self.reservation_value_utility,
            "aspiration_utility": self.aspiration_utility,
            "range": list(self.range), "provenance": self.provenance,
            "batna_note": self.batna_note,
        }


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def derive_from_view(view: PartyView, aspiration_utility: float = 0.85) -> DeclaredUtility:
    """Extract the declared utility from the party's own corpus."""
    issue = view.domain.issues[0]
    if len(view.domain.issues) > 1:
        raise NotImplementedError(
            "this deterministic derivation covers single-issue cases; with more issues the "
            "LLM planner is required (task 2.2)")
    direction = "increasing" if view.side == "seller" else "decreasing"
    wanted = "min" if direction == "increasing" else "max"

    candidates: list[tuple[float, str]] = []
    for doc in view.own_docs:
        for sentence in _sentences(doc.text):
            lowered = sentence.lower()
            hint = next((kind for phrase, kind in LIMIT_HINTS if phrase in lowered), None)
            if hint != wanted:
                continue
            for raw in MONEY.findall(sentence):
                candidates.append((_to_float(raw), sentence))
    if not candidates:
        raise ValueError(f"no declared reservation value found in the corpus of {view.role_id}")

    # A seller keeps the highest declared minimum; a buyer, the lowest declared maximum.
    value, sentence = (max(candidates) if wanted == "min" else min(candidates))
    low, high = float(issue.low), float(issue.high)
    span = max(high - low, 1e-9)
    rv_utility = (value - low) / span if direction == "increasing" else (high - value) / span
    return DeclaredUtility(
        role_id=view.role_id, issue_id=issue.issue_id, direction=direction,
        reservation_value_raw=value, reservation_value_utility=float(rv_utility),
        aspiration_utility=aspiration_utility, range=(low, high),
        provenance=[f"{view.own_docs[0].doc_id}: «{sentence[:200]}»"],
        batna_note=None,
    )
