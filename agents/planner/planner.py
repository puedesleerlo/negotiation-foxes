"""Party planner (task 2.2): writes the strategy before the first round.

What the model decides and what it does not:

  * **decides**: which foxes and tools to use and why, how to read the case, which hypotheses
    to register, where to anchor and how to concede;
  * **does not decide**: any number about the counterpart — the foxes estimate those — nor its
    own reservation value without quoting it from the corpus.

Everything it produces is **validated** before being accepted: fox ids must exist and be
usable, quotes must appear verbatim in its own corpus, and the reservation value must fall
inside the case's space. A strategy that does not validate is not used.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

import yaml

from agents.llm import LLMClient, LLMResponse
from foxes.registry import FoxRegistry
from gym.case import PartyView

PROMPTS = Path("agents/prompts")
CATALOG_TOOLS = Path("catalog/tools.yaml")


@dataclass
class PlannerResult:
    role_id: str
    strategy: dict
    raw_text: str
    reasoning: str
    usage: dict
    validation: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.validation.get("errors")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _quote_appears(quote: str, corpus: str, threshold: float = 0.82) -> bool:
    """Check the quote with tolerance: the model may trim it or normalise whitespace."""
    q, c = _normalize(quote), _normalize(corpus)
    if len(q) < 12:
        return False
    if q in c:
        return True
    window = len(q)
    for start in range(0, max(len(c) - window, 0) + 1, max(window // 3, 1)):
        chunk = c[start:start + window + 20]
        if SequenceMatcher(None, q, chunk).ratio() >= threshold:
            return True
    return False


class PartyPlanner:
    def __init__(self, view: PartyView, client: Optional[LLMClient] = None,
                 registry: Optional[FoxRegistry] = None) -> None:
        self.view = view
        self.client = client or LLMClient()
        self.registry = registry or FoxRegistry()
        self.tools = {t["tool_id"]: t for t in
                      (yaml.safe_load(CATALOG_TOOLS.read_text()) or {}).get("tools", [])}

    # ------------------------------------------------------------------ context
    def catalog_digest(self) -> str:
        """Which foxes and tools exist, with their scope. Usable ones only."""
        lines = ["### Available foxes", ""]
        for fox_id, entry in sorted(self.registry.catalog.items()):
            status = entry.get("status")
            if status == "candidate":
                continue
            scope = "; ".join(entry.get("scope_conditions") or [])
            note = " (experimental: its output does not enter the pool)" if status != "validated" else ""
            lines.append(f"- `{fox_id}`{note} — estimates {entry.get('estimates')} in phase "
                         f"{entry.get('phase')}. Scope: {scope}")
        lines += ["", "### Available tools", ""]
        for tool_id, entry in sorted(self.tools.items()):
            if entry.get("status") != "implemented":
                continue
            lines.append(f"- `{tool_id}` — {entry.get('description', '').strip()}")
        blocked = [f"`{fid}`" for fid, e in self.registry.catalog.items()
                   if e.get("status") == "candidate"]
        if blocked:
            lines += ["", f"Not available (status `candidate`, you cannot use them): "
                          f"{', '.join(sorted(blocked))}."]
        return "\n".join(lines)

    def user_prompt(self) -> str:
        issues = "\n".join(
            f"- `{spec['id']}` ({spec['type']}): "
            + (f"range {spec['range']} in {spec.get('unit', '')}, step {spec.get('step')}"
               if spec["type"] == "continuous" else f"values {spec['values']}")
            for spec in self.view.issues_spec)
        public = "\n\n".join(d.text for d in self.view.public_docs)
        own = "\n\n".join(d.text for d in self.view.own_docs)
        protocol = self.view.protocol
        return f"""# Your role

You are **{self.view.role_id}** ({self.view.side}) negotiating with **{self.view.counterpart_role_id}**.

# Protocol

Alternating offers, at most {protocol.get('max_rounds')} rounds. Free text:
{'yes' if protocol.get('allow_free_text') else 'no'}. Multiple offers:
{'yes' if protocol.get('allow_multiple_offers') else 'no'}.

# Negotiation space

{issues}

# Public information about the case

{public}

# Your confidential material

{own}

# Catalog

{self.catalog_digest()}

Now produce the strategy JSON."""

    # ------------------------------------------------------------------ execution
    def plan(self, max_tokens: int = 20000) -> PlannerResult:
        """`max_tokens` is generous on purpose: kimi-k3 spends 6-10k tokens reasoning before
        writing the JSON, and a short budget truncates it mid-structure."""
        system = (PROMPTS / "planner_system.md").read_text(encoding="utf-8")
        response: LLMResponse = self.client.complete(
            [{"role": "user", "content": self.user_prompt()}],
            system=system, max_tokens=max_tokens,
            response_format={"type": "json_object"})
        try:
            strategy = response.json()
        except json.JSONDecodeError as exc:
            return PlannerResult(
                self.view.role_id, {}, response.text, response.reasoning,
                usage=vars(response.usage),
                validation={"errors": [
                    f"the output is not valid JSON: {exc}",
                    f"(finish_reason={response.finish_reason}, "
                    f"{response.usage.completion_tokens} output tokens of which "
                    f"{response.usage.reasoning_tokens} reasoning)"]})
        result = PlannerResult(self.view.role_id, strategy, response.text, response.reasoning,
                               usage=vars(response.usage))
        result.validation = self.validate(strategy)
        return result

    # ------------------------------------------------------------------ validation
    def validate(self, strategy: dict) -> dict:
        errors: list[str] = []
        warnings: list[str] = []
        corpus = "\n".join(d.text for d in self.view.documents())

        for key in ("case_view", "declared_utility", "strategy", "plan"):
            if key not in strategy:
                errors.append(f"missing mandatory section '{key}'")
        if errors:
            return {"errors": errors, "warnings": warnings}

        # 1. The reservation value is quoted and falls inside the case's space.
        declared = strategy.get("declared_utility", {})
        rv = (declared.get("reservation_value") or {})
        issue = self.view.domain.issues[0]
        value = rv.get("value")
        if not isinstance(value, (int, float)):
            errors.append("declared_utility.reservation_value.value is not a number")
        elif not (issue.low <= float(value) <= issue.high):
            errors.append(f"the reservation value {value} is outside the case range "
                          f"[{issue.low}, {issue.high}]")
        quote = rv.get("quote", "")
        if not _quote_appears(quote, corpus):
            errors.append("the reservation value's quote does not appear in your corpus: "
                          f"«{str(quote)[:80]}»")

        # 2. The plan's foxes and tools exist and are usable.
        for task in strategy.get("plan", []):
            ident = task.get("fox_or_tool", "")
            if ident in self.registry.catalog:
                status = self.registry.status(ident)
                if status == "candidate":
                    errors.append(f"task {task.get('task_id')} uses `{ident}`, which is "
                                  f"declared `candidate` and cannot be used")
                elif status != "validated":
                    warnings.append(f"`{ident}` is experimental: its output does not enter the pool")
            elif ident in self.tools:
                if self.tools[ident].get("status") != "implemented":
                    errors.append(f"the tool `{ident}` is not implemented")
            else:
                errors.append(f"task {task.get('task_id')} names `{ident}`, which is not in "
                              f"the catalog")

        # 3. Scope coherence, checked against the *scope conditions* each fox declares, not
        # against what it can estimate. f06 can produce a prior over weights, but its scope
        # is "always applicable": it is the control and is never surplus.
        if len(self.view.domain.issues) == 1:
            for task in strategy.get("plan", []):
                ident = task.get("fox_or_tool", "")
                entry = self.registry.catalog.get(ident, {})
                scope = " ".join(entry.get("scope_conditions") or []).lower()
                if "multi-issue" in scope or ">= 2 issues" in scope:
                    errors.append(f"`{ident}` requires a multi-issue domain and this case has "
                                  f"a single issue: it is out of scope")

        # 4. Hypotheses: falsifiable, with a test.
        for i, hyp in enumerate(strategy.get("hypotheses", []) or [], 1):
            if not (hyp.get("test") or {}).get("resolves_if"):
                errors.append(f"hypothesis {i} does not say how it resolves: without a test it "
                              f"is not a hypothesis")

        return {"errors": errors, "warnings": warnings,
                "n_plan_tasks": len(strategy.get("plan", [])),
                "n_hypotheses": len(strategy.get("hypotheses", []) or []),
                "n_rejected_foxes": len(strategy.get("foxes_rejected", []) or [])}
