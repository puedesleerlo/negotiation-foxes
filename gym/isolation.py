"""Isolation between parties: canaries and runtime verification (task 2.1).

The canary test (§15) demands that a unique string from B's corpus never appears in A's
artifacts, memos, calls or indexes — and vice versa. This module provides both halves:

  * `canaries_of` — where each canary comes from;
  * `assert_no_leak` / `LeakDetector` — the check, which the gym runs on **every** artifact
    before writing it, not only in tests.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from gym.case import CaseBundle, PartyView

CANARY_RE = re.compile(r"CANARY-[A-Z0-9_]+-[0-9a-f]{6,}")


class CorpusLeak(AssertionError):
    """A corpus leak was detected: a hard failure, never a warning."""


def canaries_of(bundle: CaseBundle) -> dict[str, str]:
    """{role_id: canary}, read from each role's own corpus."""
    out: dict[str, str] = {}
    for role_id in bundle.role_ids():
        canary = bundle.party_view(role_id).canary()
        if canary:
            out[role_id] = canary
    return out


def _walk(value: Any) -> Iterable[str]:
    """Flatten any artifact (text, dict, list, dataclass) into strings."""
    if value is None:
        return
    if isinstance(value, str):
        yield value
    elif isinstance(value, Path):
        yield str(value)
        if value.exists() and value.is_file():
            try:
                yield value.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                pass
    elif isinstance(value, dict):
        for k, v in value.items():
            yield str(k)
            yield from _walk(v)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _walk(item)
    elif hasattr(value, "__dict__"):
        yield from _walk(vars(value))
    else:
        yield str(value)


def assert_no_leak(artifact: Any, *, owner_role: str, canaries: dict[str, str],
                   context: str = "") -> None:
    """Fail if an artifact of `owner_role` contains any other party's canary."""
    foreign = {role: canary for role, canary in canaries.items() if role != owner_role}
    if not foreign:
        return
    for chunk in _walk(artifact):
        for role, canary in foreign.items():
            if canary in chunk:
                where = f" in {context}" if context else ""
                raise CorpusLeak(
                    f"corpus leak{where}: an artifact of '{owner_role}' contains the canary "
                    f"of '{role}' ({canary}). Isolation is broken; the program is not valid.")


@dataclass
class LeakDetector:
    """Stateful wrapper so the gym can check every write."""

    canaries: dict[str, str]
    checks: int = 0
    findings: list[str] = field(default_factory=list)

    @classmethod
    def for_case(cls, bundle: CaseBundle) -> "LeakDetector":
        return cls(canaries=canaries_of(bundle))

    def check(self, artifact: Any, *, owner_role: str, context: str = "") -> None:
        self.checks += 1
        try:
            assert_no_leak(artifact, owner_role=owner_role, canaries=self.canaries,
                           context=context)
        except CorpusLeak as exc:
            self.findings.append(str(exc))
            raise

    def unknown_canaries(self, text: str) -> set[str]:
        """Canaries present in a text that belong to no known role."""
        return set(CANARY_RE.findall(text)) - set(self.canaries.values())

    def summary(self) -> dict:
        return {"checks": self.checks, "leaks": len(self.findings),
                "roles": sorted(self.canaries)}


def view_contains_only_own(view: PartyView, canaries: dict[str, str]) -> bool:
    """The structural check: a party's view carries only its own canary."""
    text = view.corpus_text()
    own = canaries.get(view.role_id)
    if own and own not in text:
        return False
    return all(canary not in text for role, canary in canaries.items() if role != view.role_id)
