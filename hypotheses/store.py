"""Hypothesis store and API (§9), per program and per party.

Isolation: each party has its own file; a `HypothesisStore` is opened for *one* party and cannot
read the other's. Feedback opens both, which is its privilege.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from hypotheses.schema import Hypothesis, HypothesisTest, Resolution, Threshold

RUNS_DIR = Path("runs")


class HypothesisStore:
    """One party's store. `party=None` is used only by feedback, to read everyone's."""

    def __init__(self, program_id: str, party: Optional[str],
                 runs_dir: Path = RUNS_DIR) -> None:
        self.program_id = program_id
        self.party = party
        self.root = runs_dir / program_id
        self._items: dict[str, Hypothesis] = {}
        self._load()

    # ------------------------------------------------------------------ persistence
    def _path(self, party: str) -> Path:
        return self.root / party / "hypotheses.jsonl"

    def _load(self) -> None:
        parties = [self.party] if self.party else [p.name for p in self.root.glob("*")
                                                   if p.is_dir()]
        for party in parties:
            path = self._path(party)
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = Hypothesis(**json.loads(line))
                    self._items[item.hypothesis_id] = item

    def save(self) -> None:
        by_party: dict[str, list[Hypothesis]] = {}
        for item in self._items.values():
            by_party.setdefault(item.party, []).append(item)
        for party, items in by_party.items():
            path = self._path(party)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as fh:
                for item in sorted(items, key=lambda h: h.hypothesis_id):
                    fh.write(item.model_dump_json() + "\n")

    # ------------------------------------------------------------------ registration
    def next_id(self, party: str) -> str:
        n = sum(1 for h in self._items.values() if h.party == party) + 1
        return f"{party[:1].upper()}-h{n:02d}"

    def register(self, *, party: str, statement: str, test: HypothesisTest,
                 type: str = "parameter", variable: str | None = None,
                 interval: tuple[float, float] | None = None,
                 threshold: Threshold | None = None,
                 prior_belief: float = 0.5, evidence: Iterable[str] = (),
                 created_at: str = "prep") -> Hypothesis:
        if self.party and party != self.party:
            raise PermissionError(
                f"this store belongs to '{self.party}': it cannot register hypotheses of '{party}'")
        item = Hypothesis(
            hypothesis_id=self.next_id(party), program_id=self.program_id, party=party,
            created_at=created_at, type=type, statement=statement, variable=variable,
            interval=interval, threshold=threshold, prior_belief=prior_belief,
            current_belief=prior_belief, evidence=list(evidence), test=test)
        self._items[item.hypothesis_id] = item
        return item

    # ------------------------------------------------------------------ updating
    def update(self, hypothesis_id: str, *, observation: str, confirmed: bool | None,
               at: str) -> Hypothesis:
        """Move the belief according to what the test said. `confirmed=None` = neutral evidence."""
        item = self._items[hypothesis_id]
        if confirmed is True:
            item.current_belief = item.test.p_true_if_confirmed
        elif confirmed is False:
            item.current_belief = item.test.p_true_if_disconfirmed
        item.updates.append({"at": at, "observation": observation, "confirmed": confirmed,
                             "belief": item.current_belief,
                             "ts": datetime.now(timezone.utc).isoformat()})
        return item

    def resolve(self, hypothesis_id: str, *, truth_value: bool, evidence: str,
                at: str = "feedback") -> Hypothesis:
        """Resolution against the revealed truth or realised events. Scores Brier."""
        item = self._items[hypothesis_id]
        item.resolution = Resolution(at=at, evidence=evidence, truth_value=truth_value)
        item.status = "supported" if truth_value else "refuted"
        item.brier = float((item.current_belief - (1.0 if truth_value else 0.0)) ** 2)
        return item

    def mark_unresolved(self, hypothesis_id: str, reason: str) -> Hypothesis:
        item = self._items[hypothesis_id]
        item.status = "unresolved"
        item.resolution = Resolution(at="feedback", evidence=reason, truth_value=None)
        return item

    # ------------------------------------------------------------------ queries
    def all(self) -> list[Hypothesis]:
        return sorted(self._items.values(), key=lambda h: h.hypothesis_id)

    def open(self) -> list[Hypothesis]:
        return [h for h in self.all() if h.status == "open"]

    def pending(self) -> list[Hypothesis]:
        """Everything without a verdict yet.

        `unresolved` is not a verdict: it means "could not score it with what I had". If the
        feedback improves (a new threshold, another criterion) it must be able to retry; only
        `supported` and `refuted` are final.
        """
        return [h for h in self.all() if h.status in ("open", "unresolved")]

    def by_priority(self, evpi: dict[str, float] | None = None) -> list[Hypothesis]:
        """Prioritise by expected value of information.

        Two factors: how much uncertainty is left in the hypothesis itself — Bernoulli entropy,
        maximal at p = 0.5 — and how much resolving its variable is worth, if EVPI is available.
        Without EVPI, entropy alone.
        """
        evpi = evpi or {}

        def priority(h: Hypothesis) -> float:
            p = min(max(h.current_belief, 1e-6), 1 - 1e-6)
            entropy = -(p * math.log(p) + (1 - p) * math.log(1 - p))
            weight = evpi.get(h.variable or "", 1.0)
            return entropy * max(weight, 1e-9)

        return sorted(self.open(), key=priority, reverse=True)

    def scores(self) -> dict:
        """Brier per party and per type, plus the resolved fraction (§15)."""
        resolved = [h for h in self.all() if h.brier is not None]
        out: dict = {"n_total": len(self.all()), "n_resolved": len(resolved),
                     "resolved_fraction": len(resolved) / max(len(self.all()), 1)}
        if resolved:
            out["brier_mean"] = sum(h.brier for h in resolved) / len(resolved)
            by_type: dict[str, list[float]] = {}
            by_party: dict[str, list[float]] = {}
            for h in resolved:
                by_type.setdefault(h.type, []).append(h.brier)
                by_party.setdefault(h.party, []).append(h.brier)
            out["brier_by_type"] = {k: sum(v) / len(v) for k, v in by_type.items()}
            out["brier_by_party"] = {k: sum(v) / len(v) for k, v in by_party.items()}
        return out


def disagreement_hypothesis(store: HypothesisStore, *, party: str, variable: str,
                            fox_a: str, median_a: float, fox_b: str, median_b: float,
                            sd_gap: float, at: str, probe_utility: float) -> Hypothesis:
    """Protocol rule 4 (§6.5): two foxes that disagree generate a hypothesis with a probe.

    The probe is an offer whose acceptance discriminates between the two medians: it sits
    between them, where one fox predicts acceptance and the other does not.
    """
    low, high = sorted((median_a, median_b))
    test = HypothesisTest(
        kind="probe_offer",
        spec={"offer_utility_to_counterpart": probe_utility,
              "discriminates": {fox_a: median_a, fox_b: median_b}},
        resolves_if=(f"if the counterpart accepts an offer that gives it {probe_utility:.3f}, "
                     f"this supports {fox_a if median_a < median_b else fox_b} (lower rv); "
                     f"if it rejects, it supports the other"),
    )
    return store.register(
        party=party, type="parameter", variable=variable,
        statement=(f"The counterpart's {variable} is below {high:.3f} "
                   f"({sd_gap:.2f} sd disagreement between {fox_a} and {fox_b})"),
        interval=(low, high), prior_belief=0.5,
        evidence=[f"fox:{fox_a}", f"fox:{fox_b}"], test=test, created_at=at)
