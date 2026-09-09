"""The negotiator's writing layer (the LLM half of task 2.7).

Deliberate separation: the **action** is chosen by `ScriptedNegotiator` with the personality's
rule and the foxes' posteriors; the model only **writes** the reasoning for the record and the
message that goes to the table. This is the operational form of protocol rules 7 and 8 (§6.5):
the LLM cannot override the foxes because it is not the one deciding.

If the model fails or there are no credentials, the turn continues with the deterministic
memo: a negotiation never stops because of a writing problem.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from agents.llm import CredentialsMissing, LLMClient
from gym.protocol import Action

PROMPTS = Path("agents/prompts")


@dataclass
class WrittenTurn:
    rationale: str
    message: Optional[str]
    disclosure_reasoning: str = ""
    expects_to_learn: str = ""
    concern: str = ""
    usage: dict = field(default_factory=dict)
    reasoning: str = ""
    failed: Optional[str] = None


def _compact_memo(memo: dict, issue_id: str) -> dict:
    """Only what is needed to write. Short prompt = fast, cheap turn."""
    belief = memo.get("belief_summary", {})
    chosen = memo.get("chosen", {})
    return {
        "round": memo.get("round"),
        "decided_action": chosen.get("action"),
        "offer": [o.get(issue_id) for o in chosen.get("offers", [])] or None,
        "belief_about_their_reservation_value": belief.get("rv"),
        "foxes_in_scope": belief.get("foxes_in_scope"),
        "foxes_out_of_scope": belief.get("foxes_out_of_scope"),
        "disagreement_between_foxes_sd": belief.get("disagreement_sd"),
        "candidates": memo.get("candidates", [])[:3],
        "concession_target": memo.get("concession_target"),
        "acceptance_check": memo.get("acceptance_check"),
        "deterministic_reasoning": memo.get("rationale"),
        "open_hypotheses": memo.get("hypotheses_in_play"),
    }


class TurnWriter:
    def __init__(self, role_id: str, side: str, counterpart_id: str, issue_id: str,
                 client: Optional[LLMClient] = None, max_tokens: int = 6000) -> None:
        self.role_id = role_id
        self.side = side
        self.counterpart_id = counterpart_id
        self.issue_id = issue_id
        self.client = client or LLMClient()
        self.max_tokens = max_tokens
        self.system = (PROMPTS / "negotiator_system.md").read_text(encoding="utf-8")

    def write(self, action: Action, memo: dict, last_message: Optional[str] = None) -> WrittenTurn:
        fallback = WrittenTurn(rationale=memo.get("rationale", ""), message=None)
        if not self.client.configured:
            fallback.failed = "no credentials: the deterministic memo is kept"
            return fallback

        payload = {
            "i_am": f"{self.role_id} ({self.side})",
            "negotiating_with": self.counterpart_id,
            "issue": self.issue_id,
            "their_last_message": last_message,
            "decision_memo": _compact_memo(memo, self.issue_id),
        }
        try:
            response = self.client.complete(
                [{"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)}],
                system=self.system, max_tokens=self.max_tokens,
                response_format={"type": "json_object"})
            data: dict[str, Any] = response.json()
        except CredentialsMissing as exc:
            fallback.failed = str(exc)
            return fallback
        except Exception as exc:   # the turn does not stop for a writing failure
            fallback.failed = f"{type(exc).__name__}: {exc}"
            return fallback

        return WrittenTurn(
            rationale=str(data.get("rationale") or memo.get("rationale", ""))[:1200],
            message=(str(data.get("message"))[:600] if data.get("message") else None),
            disclosure_reasoning=str(data.get("disclosure_reasoning") or "")[:600],
            expects_to_learn=str(data.get("expects_to_learn") or "")[:600],
            concern=str(data.get("concern") or "")[:600],
            usage=vars(response.usage), reasoning=response.reasoning)
