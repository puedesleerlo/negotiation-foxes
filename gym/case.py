"""Case loading with strict corpus isolation (task 2.1).

Non-negotiable principle (§3): the agent of one party never receives text, indexes or traces
of the other. Here that is not a convention but a structural property: `PartyView` only holds
paths of the public corpus and of its own role, and keeps no reference to the rest of the case.

The **sealed truth** lives in `SealedTruth`, obtained through a separate, explicit method
(`CaseBundle.sealed_truth`) so its use is visible in code review: only `feedback/` should call it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml

from foxes.domain import Domain, Issue, Utility

CASES_DIR = Path("cases")


@dataclass(frozen=True)
class Document:
    doc_id: str
    path: Path
    visibility: str          # public | confidential
    role_id: str | None
    text: str

    def excerpt(self, max_chars: int = 400) -> str:
        return self.text[:max_chars]


@dataclass(frozen=True)
class PartyView:
    """Everything a party may read. Contains nothing of the counterpart, by construction."""

    case_id: str
    role_id: str
    side: str
    public_docs: tuple[Document, ...]
    own_docs: tuple[Document, ...]
    domain: Domain
    protocol: dict
    counterpart_role_id: str          # only the identifier: to address them at the table
    issues_spec: tuple[dict, ...]

    def documents(self) -> Iterator[Document]:
        yield from self.public_docs
        yield from self.own_docs

    def corpus_text(self) -> str:
        return "\n\n".join(d.text for d in self.documents())

    def canary(self) -> str | None:
        for doc in self.own_docs:
            for line in doc.text.splitlines():
                if line.strip().startswith("canary:"):
                    return line.split(":", 1)[1].strip()
        return None


@dataclass
class SealedTruth:
    """The frozen truth of the case. Read by feedback, by nobody else."""

    case_id: str
    per_role: dict[str, dict]
    policy_applied: str

    def reservation_value(self, role_id: str) -> float:
        return float(self.per_role[role_id]["reservation_value"]["value"])

    def reservation_value_utility(self, role_id: str) -> float:
        return float(self.per_role[role_id]["utility"]["reservation_value_utility"])

    def true_zopa(self, seller: str, buyer: str) -> tuple[float, float]:
        return self.reservation_value(seller), self.reservation_value(buyer)


@dataclass
class CaseBundle:
    case_id: str
    root: Path
    spec: dict
    _docs: dict[str, list[Document]] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, case_id: str, cases_dir: Path = CASES_DIR) -> "CaseBundle":
        root = cases_dir / case_id
        spec = yaml.safe_load((root / "case.yaml").read_text(encoding="utf-8"))
        bundle = cls(case_id=case_id, root=root, spec=spec)
        bundle._load_documents()
        return bundle

    def _load_documents(self) -> None:
        self._docs = {"public": []}
        public_dir = self.root / self.spec.get("public_corpus", "public/")
        for path in sorted(public_dir.glob("**/*.md")):
            self._docs["public"].append(Document(
                doc_id=f"public/{path.name}", path=path, visibility="public",
                role_id=None, text=path.read_text(encoding="utf-8")))
        for role in self.spec["roles"]:
            role_id = role["id"]
            self._docs[role_id] = []
            role_dir = self.root / role["corpus"]
            for path in sorted(role_dir.glob("**/*.md")):
                self._docs[role_id].append(Document(
                    doc_id=f"{role_id}/{path.name}", path=path, visibility="confidential",
                    role_id=role_id, text=path.read_text(encoding="utf-8")))

    # ------------------------------------------------------------------ domain
    def domain(self) -> Domain:
        issues = []
        for spec in self.spec["issues"]:
            if spec["type"] == "continuous":
                low, high = spec["range"]
                step = spec.get("step")
                steps = int(round((high - low) / step)) + 1 if step else 21
                issues.append(Issue(spec["id"], "continuous", low=float(low), high=float(high),
                                    steps=steps))
            else:
                issues.append(Issue(spec["id"], "discrete", values=tuple(spec["values"])))
        return Domain(self.case_id, tuple(issues))

    def role_ids(self) -> list[str]:
        return [r["id"] for r in self.spec["roles"]]

    def counterpart_of(self, role_id: str) -> str:
        others = [r for r in self.role_ids() if r != role_id]
        if len(others) != 1:
            raise ValueError("E1 assumes exactly two parties")
        return others[0]

    # ------------------------------------------------------------------ views
    def party_view(self, role_id: str) -> PartyView:
        """The only way to hand material to a party agent."""
        if role_id not in self.role_ids():
            raise KeyError(f"{role_id} is not a role of {self.case_id}")
        role = next(r for r in self.spec["roles"] if r["id"] == role_id)
        return PartyView(
            case_id=self.case_id, role_id=role_id, side=role.get("side", "?"),
            public_docs=tuple(self._docs["public"]),
            own_docs=tuple(self._docs[role_id]),
            domain=self.domain(), protocol=dict(self.spec["protocol"]),
            counterpart_role_id=self.counterpart_of(role_id),
            issues_spec=tuple(self.spec["issues"]),
        )

    # ------------------------------------------------------------------ sealed truth
    def sealed_truth(self) -> SealedTruth:
        """For `feedback/` ONLY. No agent or fox may call this method."""
        raw = self.spec["sealed_truth"]
        per_role = {rid: raw[rid] for rid in self.role_ids() if rid in raw}
        return SealedTruth(case_id=self.case_id, per_role=per_role,
                           policy_applied=raw.get("policy_applied", "declared"))

    def true_utility(self, role_id: str) -> Utility:
        """A role's true utility, derived from the sealed truth. Feedback only."""
        truth = self.sealed_truth().per_role[role_id]
        domain = self.domain()
        spec = self.spec["issues"][0]
        low, high = spec["range"]
        increasing = truth["utility"]["direction"] == "increasing"
        value_maps: dict[str, Any] = {
            spec["id"]: (float(low), float(high)) if increasing else (float(high), float(low))}
        util = Utility(domain, {spec["id"]: 1.0}, value_maps)
        util.reservation_value = float(truth["utility"]["reservation_value_utility"])
        return util
