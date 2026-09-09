"""Integrative tools: creating value instead of only dividing it (task 2.8).

  t06 `generate_mesos`      — multiple equivalent simultaneous offers as a probe
  t07 `find_logrolling`     — trades that raise expected joint utility
  t08 `build_contingent`    — a clause that turns a difference in beliefs into value

All three require more than one issue (or an uncertain event, for the contingent clause). In a
single-issue distributive case they are **not applicable**, and they say so: they return
`NotApplicable` with the reason, instead of producing something that looks like an integrative
option where none exists.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from foxes.domain import Domain, Offer, Utility


@dataclass
class NotApplicable:
    tool_id: str
    reason: str
    applicable: bool = False

    def __bool__(self) -> bool:  # so the agent can write `if result:`
        return False


@dataclass
class MesoSet:
    tool_id: str
    offers: list[Offer]
    target_utility: float
    spread: float                     # dispersion in issue space: the more, the more informative
    applicable: bool = True
    detail: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.offers)


def _issue_positions(domain: Domain, offer: Offer) -> np.ndarray:
    """Normalised position [0,1] of the offer on each issue, to measure spread."""
    out = []
    for issue in domain.issues:
        if issue.type == "continuous":
            span = max(issue.high - issue.low, 1e-9)
            out.append((float(offer[issue.issue_id]) - issue.low) / span)
        else:
            values = list(issue.values)
            out.append(values.index(offer[issue.issue_id]) / max(len(values) - 1, 1))
    return np.array(out)


def generate_mesos(domain: Domain, own_utility: Utility, target_utility: float,
                   k: int = 3, tolerance: float = 0.02) -> MesoSet | NotApplicable:
    """t06 — k packages of equal own utility and maximal dispersion among them.

    Equivalence is what makes them honest (the party is indifferent to which one is accepted)
    and dispersion is what makes them informative: the counterpart's choice reveals which issue
    matters to it. With a single issue, "equivalent" implies "identical" and nothing is revealed.
    """
    if len(domain.issues) < 2:
        return NotApplicable("t06_meso_generator",
                             "a MESO needs at least two issues: with a single one, two offers "
                             "of the same own utility are the same offer")
    space = domain.outcome_space()
    values = np.array([own_utility(o) for o in space])
    band = [space[i] for i in np.flatnonzero(np.abs(values - target_utility) <= tolerance)]
    if len(band) < k:
        order = np.argsort(np.abs(values - target_utility))[: max(k * 4, 12)]
        band = [space[i] for i in order]
    positions = np.array([_issue_positions(domain, o) for o in band])

    # Greedy maximum-dispersion selection (k-centres): start with the most separated pair.
    if len(band) <= k:
        chosen = list(range(len(band)))
    else:
        best_pair = max(combinations(range(len(band)), 2),
                        key=lambda p: np.linalg.norm(positions[p[0]] - positions[p[1]]))
        chosen = list(best_pair)
        while len(chosen) < k:
            remaining = [i for i in range(len(band)) if i not in chosen]
            nxt = max(remaining, key=lambda i: min(np.linalg.norm(positions[i] - positions[j])
                                                   for j in chosen))
            chosen.append(nxt)
    offers = [band[i] for i in chosen]
    spread = float(np.mean([np.linalg.norm(positions[a] - positions[b])
                            for a, b in combinations(chosen, 2)])) if len(chosen) > 1 else 0.0
    utilities = [float(own_utility(o)) for o in offers]
    return MesoSet(tool_id="t06_meso_generator", offers=offers,
                   target_utility=float(np.mean(utilities)), spread=spread,
                   detail={"utilities": utilities,
                           "max_utility_gap": float(max(utilities) - min(utilities))})


@dataclass
class Trade:
    give_issue: str
    take_issue: str
    from_offer: Offer
    to_offer: Offer
    own_delta: float
    expected_other_delta: float
    expected_joint_delta: float


def find_logrolling(domain: Domain, own_utility: Utility, base_offer: Offer,
                    other_weight_samples: np.ndarray, other_directions: np.ndarray,
                    max_results: int = 5) -> list[Trade] | NotApplicable:
    """t07 — Trades that raise the expected joint utility under the belief.

    Looks for moves that give on one issue and gain on another, ranked by expected joint gain.
    It requires a belief over the counterpart's weights: if f02/f09 are out of scope, this tool
    has no valid input.
    """
    if len(domain.issues) < 2:
        return NotApplicable("t07_logrolling_finder",
                             "logrolling trades across issues: with a single one it does not exist")
    if other_weight_samples.size == 0:
        return NotApplicable("t07_logrolling_finder",
                             "without samples of the counterpart's weights there is nothing to "
                             "compare; f02 and f09 must be in scope")

    from foxes.f07_zopa_pareto_estimator.fox import utility_from_theta

    others = [utility_from_theta(domain, other_weight_samples[i], other_directions[i], 0.0)
              for i in range(min(len(other_weight_samples), 120))]
    base_own = float(own_utility(base_offer))
    base_other = float(np.mean([u(base_offer) for u in others]))

    trades: list[Trade] = []
    for offer in domain.outcome_space():
        changed = [i.issue_id for i in domain.issues if offer[i.issue_id] != base_offer[i.issue_id]]
        if not 1 <= len(changed) <= 2:
            continue
        own_delta = float(own_utility(offer)) - base_own
        other_delta = float(np.mean([u(offer) for u in others])) - base_other
        joint = own_delta + other_delta
        if joint <= 1e-9:
            continue
        give = next((i for i in changed if own_delta < 0), changed[0])
        take = next((i for i in changed if i != give), changed[-1])
        trades.append(Trade(give_issue=give, take_issue=take, from_offer=base_offer,
                            to_offer=offer, own_delta=own_delta,
                            expected_other_delta=other_delta, expected_joint_delta=joint))
    trades.sort(key=lambda t: t.expected_joint_delta, reverse=True)
    return trades[:max_results]


@dataclass
class ContingentClause:
    tool_id: str
    event: str
    own_probability: float
    counterpart_probability: float
    payment_if_event: float
    payment_if_not: float
    own_expected_value: float
    counterpart_expected_value: float
    applicable: bool = True

    def __bool__(self) -> bool:
        return self.own_expected_value > 0 and self.counterpart_expected_value > 0


def build_contingent(event: str, own_probability: float, counterpart_probability: float,
                     stake: float, min_gap: float = 0.10) -> ContingentClause | NotApplicable:
    """t08 — Contingent clause: bet on the difference in beliefs instead of arguing it.

    If I believe the event is more likely than you do, I accept being paid more if it happens
    in exchange for being paid less if it does not. Under *each* belief the deal has positive
    expected value for both: the difference of opinion becomes value instead of an obstacle.

    Honesty notice: the corpus (L2, section 3) says explicitly that contingent contracts are
    mentioned but have **no** computational treatment or empirical quantification in the
    retrieved literature. This tool implements the mutually-favourable-bet criterion; there is
    no empirical evidence behind its effect in real negotiation.
    """
    gap = own_probability - counterpart_probability
    if abs(gap) < min_gap:
        return NotApplicable("t08_contingent_builder",
                             f"beliefs about '{event}' differ by {abs(gap):.2f}, below the "
                             f"minimum {min_gap}: no difference to exploit")
    # The bet is priced at the *midpoint* of the two beliefs. Pricing it at one's own belief
    # would give oneself zero expected value: the surplus comes precisely from the difference,
    # and the midpoint splits it evenly (stake·|gap|/2 to each side).
    q = (own_probability + counterpart_probability) / 2.0
    direction = 1.0 if gap > 0 else -1.0          # +1: I collect if it happens; −1: if it does not
    pay_event = direction * stake * (1 - q)
    pay_not = -direction * stake * q
    own_ev = own_probability * pay_event + (1 - own_probability) * pay_not
    cp_ev = -(counterpart_probability * pay_event + (1 - counterpart_probability) * pay_not)
    return ContingentClause(
        tool_id="t08_contingent_builder", event=event, own_probability=own_probability,
        counterpart_probability=counterpart_probability, payment_if_event=pay_event,
        payment_if_not=pay_not, own_expected_value=own_ev, counterpart_expected_value=cp_ev)
