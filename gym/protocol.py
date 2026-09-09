"""Alternating-offers protocol and action validation (task 3.1).

The gym validates **every** action against `case.yaml` before accepting it: an offer out of
range, an issue that does not exist or a MESO where the case forbids it are rejected and
recorded as invalid. The agent cannot step outside the case's declared space.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from foxes.domain import Domain, Offer

ActionType = Literal["propose", "propose_multiple", "accept", "reject", "walk_away",
                     "message_only"]


@dataclass
class Action:
    round: int
    party: str
    action: ActionType
    offers: list[Offer] = field(default_factory=list)
    message: Optional[str] = None
    contingent_terms: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def offer(self) -> Optional[Offer]:
        return self.offers[0] if self.offers else None


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    normalized: Optional[Action] = None


class Protocol:
    def __init__(self, domain: Domain, spec: dict, issues_spec: list[dict]) -> None:
        self.domain = domain
        self.spec = spec
        self.issues_spec = {i["id"]: i for i in issues_spec}
        self.max_rounds = int(spec.get("max_rounds", 20))
        self.allow_free_text = bool(spec.get("allow_free_text", True))
        self.allow_multiple = bool(spec.get("allow_multiple_offers", False))

    # ------------------------------------------------------------------ validation
    def validate_offer(self, offer: Offer) -> list[str]:
        errors: list[str] = []
        expected = set(self.domain.issue_ids)
        got = set(offer)
        if got != expected:
            errors.append(f"the offer must set exactly {sorted(expected)}, it sets {sorted(got)}")
            return errors
        for issue in self.domain.issues:
            value = offer[issue.issue_id]
            if issue.type == "continuous":
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    errors.append(f"{issue.issue_id}: '{value}' is not a number")
                    continue
                if not (issue.low - 1e-9 <= number <= issue.high + 1e-9):
                    errors.append(f"{issue.issue_id}={number} is outside the range "
                                  f"[{issue.low}, {issue.high}]")
                step = self.issues_spec.get(issue.issue_id, {}).get("step")
                if step:
                    k = round((number - issue.low) / step)
                    if abs(issue.low + k * step - number) > 1e-6:
                        errors.append(f"{issue.issue_id}={number} is not on the step of {step}")
            else:
                if value not in issue.values:
                    errors.append(f"{issue.issue_id}='{value}' is not among {list(issue.values)}")
        return errors

    def validate(self, action: Action) -> ValidationResult:
        errors: list[str] = []
        if action.round < 1 or action.round > self.max_rounds:
            errors.append(f"round {action.round} is outside the horizon [1, {self.max_rounds}]")
        if action.action == "propose_multiple" and not self.allow_multiple:
            errors.append("this case does not allow multiple offers (allow_multiple_offers: false)")
        if action.message and not self.allow_free_text:
            errors.append("this case does not allow free text")
        if action.action in ("propose", "propose_multiple"):
            if not action.offers:
                errors.append(f"the action '{action.action}' requires at least one offer")
            if action.action == "propose" and len(action.offers) > 1:
                errors.append("'propose' carries exactly one offer; use 'propose_multiple'")
            for offer in action.offers:
                errors.extend(self.validate_offer(offer))
        elif action.action in ("accept", "reject", "walk_away", "message_only"):
            if action.offers:
                errors.append(f"the action '{action.action}' carries no offers")
        else:
            errors.append(f"unknown action: {action.action}")
        if action.contingent_terms and not self.spec.get("allow_contingent_terms", False):
            errors.append("this case does not allow contingent terms")
        return ValidationResult(ok=not errors, errors=errors,
                                normalized=action if not errors else None)

    # ------------------------------------------------------------------ dynamics
    def first_mover(self, role_ids: list[str], rng) -> str:
        setting = self.spec.get("first_mover", "random")
        if setting in role_ids:
            return setting
        return role_ids[int(rng.integers(len(role_ids)))]

    def is_terminal(self, action: Action) -> bool:
        return action.action in ("accept", "walk_away")

    def deadline_reached(self, round_idx: int) -> bool:
        return round_idx > self.max_rounds
