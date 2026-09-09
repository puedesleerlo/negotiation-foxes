"""Deterministic negotiator: the party that decides without an LLM (task 2.7, model-free core).

It does everything the LLM-backed negotiator does except write prose: observe the table, update
the foxes, pool their posteriors, register hypotheses when foxes disagree, score candidates with
the personality's rule and write a `decision_memo` every turn.

It exists for two reasons: it lets complete programs run without credentials, and it keeps the
decision core separate from the text. The LLM (`TurnWriter`) replaces the wording of the memo
and the table message, never the choice of action — which must stay auditable (§6.5, rules 7
and 8).

Which foxes run online is decided by the planner's plan (`online_foxes`), not by this class:
the strategy is the contract (§7.3). Without a plan, a default set is used and recorded as such.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from agents.declared_utility import DeclaredUtility
from agents.personality import Candidate, Personality
from foxes.base import Observation
from foxes.pooling import disagreement, linear_pool
from foxes.registry import FoxRegistry
from gym.case import PartyView
from gym.protocol import Action, Protocol
from hypotheses.schema import Hypothesis
from hypotheses.store import HypothesisStore, disagreement_hypothesis
from hypotheses.thresholds import threshold_from_prose
from tools.decision import evpi

DEFAULT_ONLINE_FOXES = ("f01_bayes_rv_concession", "f03_time_concession_regression")
CONTROL_FOX = "f06_uninformed_prior"


@dataclass
class Beliefs:
    theta: dict[str, np.ndarray]
    per_fox: dict[str, Any]
    pooled_rv: Optional[Any]
    in_scope: list[str]           # validated foxes whose output entered the pool
    out_of_scope: list[str]       # ran, but excluded: out of scope, experimental, or the control
    disagreement_sd: float = 0.0
    summary: dict = field(default_factory=dict)


def counterpart_utility_model(view: PartyView, own_utility) -> Callable[[dict], float]:
    """How this party estimates what an offer is worth *to the counterpart*.

    Single-issue distributive case: the two linear utilities are complementary, so the estimate
    is exactly `1 - u_own`. That is the only case E1 currently supports; for a multi-issue case
    the estimate must come from the weight foxes (f02/f09) and that wiring does not exist yet.
    Failing loudly here is deliberate — a silently wrong estimate would corrupt every fox that
    consumes it (f01, f03) without any symptom in the trace.
    """
    if len(view.domain.issues) == 1:
        return lambda offer: 1.0 - float(own_utility(offer))
    raise NotImplementedError(
        f"case {view.case_id} has {len(view.domain.issues)} issues: the counterpart utility "
        f"estimate must be built from the f02/f09 weight posteriors, which is not wired yet "
        f"(see REVIEW.md, finding F4)")


class ScriptedNegotiator:
    def __init__(self, view: PartyView, declared: DeclaredUtility, personality: Personality,
                 protocol: Protocol, hypotheses: HypothesisStore, *, program_id: str,
                 registry: Optional[FoxRegistry] = None, seed: int = 0,
                 n_samples: int = 2000, online_foxes: Optional[list[str]] = None) -> None:
        self.view = view
        self.declared = declared
        self.utility = declared.to_utility(view)
        self.personality = personality
        self.protocol = protocol
        self.hypotheses = hypotheses
        self.program_id = program_id
        self.registry = registry or FoxRegistry()
        self.rng = np.random.default_rng(seed)
        self.n_samples = n_samples
        self.space = view.domain.outcome_space()
        self.estimate_counterpart_utility = counterpart_utility_model(view, self.utility)

        self.last_beliefs: Optional[Beliefs] = None
        self.aspiration: Optional[float] = None   # set in preparation, from the belief
        self.strategy: Optional[dict] = None      # the planner's strategy, if there was one
        self.first_offer: Optional[float] = None
        self.my_last_offer: Optional[dict] = None
        self.foxes_source = "plan" if online_foxes is not None else "default"
        self.foxes: dict[str, Any] = {}
        self.states: dict[str, Any] = {}
        self.experimental: set[str] = set()

        wanted = list(online_foxes) if online_foxes is not None else list(DEFAULT_ONLINE_FOXES)
        if CONTROL_FOX not in wanted:
            wanted.append(CONTROL_FOX)     # the control runs in every program (§6.5, rule 2)
        for fox_id in wanted:
            entry = self.registry.catalog.get(fox_id)
            if entry is None or entry.get("status") == "candidate":
                continue                   # a candidate cannot be instantiated; skip silently here,
                                           # the plan validation already refused it upstream
            experimental = entry.get("status") != "validated"
            kwargs = {"params": ["rv"]} if fox_id == CONTROL_FOX else {}
            fox = self.registry.create(fox_id, allow_experimental=experimental, **kwargs)
            state = fox.init_state(view.domain, self.utility)
            state.data["counterpart_party"] = view.counterpart_role_id
            state.program_id, state.party = program_id, view.role_id
            self.foxes[fox_id] = fox
            self.states[fox_id] = state
            if experimental:
                self.experimental.add(fox_id)

    # ------------------------------------------------------------------ observation
    def observe(self, action: Action) -> None:
        """Feed the foxes and the hypotheses with what the counterpart put on the table.

        A counter-offer carries two signals: the offer itself, and the implied rejection of
        this party's previous offer. An acceptance carries the acceptance of that offer. Both
        are emitted, so acceptance-boundary foxes (f04) see real responses.
        """
        if action.party == self.view.role_id:
            return
        round_idx, max_rounds = action.round, self.protocol.max_rounds

        if self.my_last_offer is not None and action.action in ("propose", "accept"):
            response = Observation(
                round=round_idx, party=action.party, offer=self.my_last_offer,
                action="accept" if action.action == "accept" else "reject",
                utility_to_observer=float(self.utility(self.my_last_offer)),
                est_utility_to_proposer=self.estimate_counterpart_utility(self.my_last_offer),
                max_rounds=max_rounds)
            for fox_id, fox in self.foxes.items():
                self.states[fox_id] = fox.update(self.states[fox_id], response)

        if action.offer:
            u_to_me = float(self.utility(action.offer))
            obs = Observation(
                round=round_idx, party=action.party, offer=action.offer, action=action.action,
                utility_to_observer=u_to_me,
                est_utility_to_proposer=self.estimate_counterpart_utility(action.offer),
                max_rounds=max_rounds)
            for fox_id, fox in self.foxes.items():
                self.states[fox_id] = fox.update(self.states[fox_id], obs)

        self.update_hypotheses(action)

    # ------------------------------------------------------------------ hypotheses
    def _counterpart_rv_bound(self, action: Action) -> Optional[tuple[str, float, float]]:
        """What a counterpart action implies about its reservation value.

        Returns (direction, bound_in_issue_units, bound_in_counterpart_utility) or None.
        A party never offers below its own reserve, so its offer is a bound on the reserve:
        a seller offering price P has rv <= P; a buyer offering P has rv >= P. Accepting this
        party's offer bounds the same way through the accepted offer.
        """
        offer = action.offer if action.action == "propose" else (
            self.my_last_offer if action.action == "accept" else None)
        if not offer:
            return None
        issue_id = self.declared.issue_id
        price = float(offer[issue_id])
        u_cp = self.estimate_counterpart_utility(offer)
        counterpart_side = "buyer" if self.view.side == "seller" else "seller"
        direction = "lte" if counterpart_side == "seller" else "gte"
        return direction, price, u_cp

    def update_hypotheses(self, action: Action) -> list[str]:
        """§5.2 step 3: mark the open hypotheses this observation supports or refutes.

        Only decisive observations move a belief; a compatible-but-uninformative one leaves it
        alone. A refutation is permanent — bounds only tighten, so nothing later can undo it.
        """
        bound = self._counterpart_rv_bound(action)
        if bound is None:
            return []
        direction, price, u_cp = bound
        touched: list[str] = []
        for item in self.hypotheses.open():
            if item.variable != "theta.rv":
                continue
            verdict = self._judge(item, direction, price, u_cp)
            if verdict is None:
                continue
            already_refuted = any(u.get("confirmed") is False for u in item.updates)
            if already_refuted or (verdict and item.current_belief == item.test.p_true_if_confirmed):
                continue
            self.hypotheses.update(
                item.hypothesis_id, confirmed=verdict, at=f"round:{action.round}",
                observation=(f"{action.party} {action.action} at {price:,.0f} "
                             f"({u_cp:.3f} in its utility) implies rv {'<=' if direction == 'lte' else '>='} "
                             f"that; the hypothesis is {'supported' if verdict else 'refuted'}"))
            touched.append(item.hypothesis_id)
        return touched

    @staticmethod
    def _judge(item: Hypothesis, direction: str, price: float, u_cp: float) -> Optional[bool]:
        """True = supported, False = refuted, None = this observation says nothing."""
        # Structured threshold in issue units (what the planner is asked to produce); if it is
        # missing, the same conservative prose parser the feedback uses — never an invented one.
        cmp = vals = None
        if item.threshold is not None and item.threshold.unit.upper() == "USD":
            cmp, vals = item.threshold.comparison, item.threshold.values
        elif item.threshold is None:
            parsed = threshold_from_prose(item.statement)
            if parsed is not None:
                cmp, vals = parsed[0], [parsed[1]]
        if cmp is not None:
            if cmp == "gte":
                return True if (direction == "gte" and price >= vals[0]) else (
                    False if (direction == "lte" and price < vals[0]) else None)
            if cmp == "lte":
                return True if (direction == "lte" and price <= vals[0]) else (
                    False if (direction == "gte" and price > vals[0]) else None)
            if cmp == "between":
                lo, hi = min(vals[:2]), max(vals[:2])
                if direction == "lte" and price < lo:
                    return False
                if direction == "gte" and price > hi:
                    return False
                return None
        # Interval in the counterpart's utility scale (the fox-disagreement hypothesis:
        # "rv is below `high`"). An offer at u_cp bounds rv <= u_cp.
        if item.interval is not None:
            _, high = item.interval
            if direction == "lte" and u_cp <= high:
                return True
            return None
        return None

    def maybe_register_hypothesis(self, beliefs: Beliefs, round_idx: int) -> Optional[str]:
        """Protocol rule 4: disagreement above the threshold -> a hypothesis with a probe."""
        if beliefs.disagreement_sd < self.personality.disagreement_threshold:
            return None
        sources = [f for f in beliefs.in_scope if "rv" in beliefs.per_fox[f].params]
        if len(sources) < 2:
            return None
        a, b = sources[0], sources[1]
        median_a = beliefs.per_fox[a].quantile("rv", 0.5)
        median_b = beliefs.per_fox[b].quantile("rv", 0.5)
        if any(h.variable == "theta.rv" and h.status == "open" and h.interval is not None
               for h in self.hypotheses.all()):
            return None
        item = disagreement_hypothesis(
            self.hypotheses, party=self.view.role_id, variable="theta.rv",
            fox_a=a, median_a=median_a, fox_b=b, median_b=median_b,
            sd_gap=beliefs.disagreement_sd, at=f"round:{round_idx}",
            probe_utility=float((median_a + median_b) / 2))
        return item.hypothesis_id

    # ------------------------------------------------------------------ belief
    def beliefs(self, round_idx: int) -> Beliefs:
        per_fox, in_scope, out_scope = {}, [], []
        for fox_id, fox in self.foxes.items():
            self.states[fox_id].round = round_idx
            post = self.registry.call(fox, self.states[fox_id], step="negotiation",
                                      experimental=fox_id in self.experimental)
            per_fox[fox_id] = post
            usable = post.scope_ok and fox_id != CONTROL_FOX and fox_id not in self.experimental
            (in_scope if usable else out_scope).append(fox_id)

        rv_sources = [per_fox[f] for f in in_scope if "rv" in per_fox[f].params]
        if rv_sources:
            pooled = linear_pool(rv_sources, "rv", n=self.n_samples,
                                 seed=int(self.rng.integers(1 << 31)))
            rv = pooled.resample(self.n_samples, self.rng)[:, 0]
            gap = disagreement(rv_sources, "rv") if len(rv_sources) > 1 else 0.0
        else:
            pooled, gap = None, 0.0
            control = per_fox.get(CONTROL_FOX)
            rv = (control.resample(self.n_samples, self.rng)[:, 0] if control
                  else self.rng.uniform(0, 1, self.n_samples))

        # beta and T come from whichever in-scope foxes estimate them (f03 today); otherwise a
        # flat prior, declared in the summary so the memo can say it.
        shape_sources = [per_fox[f] for f in in_scope
                         if "beta" in per_fox[f].params and "T" in per_fox[f].params]
        if shape_sources:
            draws = shape_sources[0].resample(self.n_samples, self.rng)
            beta = draws[:, shape_sources[0].params.index("beta")]
            deadline = draws[:, shape_sources[0].params.index("T")]
            shape_from = shape_sources[0].fox_id
        else:
            beta = self.rng.uniform(0.2, 3.0, self.n_samples)
            deadline = self.rng.uniform(10, 30, self.n_samples)
            shape_from = "flat prior"

        theta = {"rv": rv, "beta": beta, "T": deadline}
        summary = {
            "rv": {"median": float(np.median(rv)),
                   "q05": float(np.quantile(rv, 0.05)), "q95": float(np.quantile(rv, 0.95))},
            "beta_median": float(np.median(beta)), "T_median": float(np.median(deadline)),
            "shape_from": shape_from,
            "foxes_in_scope": in_scope, "foxes_out_of_scope": out_scope,
            "foxes_source": self.foxes_source,
            "disagreement_sd": gap,
        }
        return Beliefs(theta=theta, per_fox=per_fox, pooled_rv=pooled, in_scope=in_scope,
                       out_of_scope=out_scope, disagreement_sd=gap, summary=summary)

    # ------------------------------------------------------------------ decision
    def decide(self, round_idx: int, offer_on_table: Optional[dict]) -> tuple[Action, dict]:
        beliefs = self.beliefs(round_idx)
        self.last_beliefs = beliefs   # the gym persists it to score calibration per round
        hypothesis_id = self.maybe_register_hypothesis(beliefs, round_idx)

        score = lambda rnd: self.personality.score_offers(          # noqa: E731
            self.space, self.utility, rnd, beliefs.theta,
            est_counterpart_utility=self.estimate_counterpart_utility)
        candidates = score(round_idx)
        target = self.personality.concession_target(
            round_idx, self.protocol.max_rounds,
            aspiration=self.aspiration if self.aspiration is not None
            else self.declared.aspiration_utility,
            reservation=self.utility.reservation_value)
        allowed = [c for c in candidates if c.u_own >= target - 1e-9] or candidates[:1]
        best = allowed[0]

        memo: dict[str, Any] = {
            "round": round_idx, "party": self.view.role_id,
            "belief_summary": beliefs.summary,
            "counterpart_utility_model": "complement (single-issue distributive)",
            "foxes_consulted": [
                {"fox_id": fid, "version": post.version, "scope_ok": post.scope_ok,
                 "in_pool": fid in beliefs.in_scope,
                 "said": {p: round(post.quantile(p, 0.5), 4) for p in post.params},
                 "warnings": post.warnings[:2]}
                for fid, post in beliefs.per_fox.items()],
            "hypotheses_in_play": [h.hypothesis_id for h in self.hypotheses.open()],
            "concession_target": target,
            "candidates": [
                {"offer": c.offer, "u_own": round(c.u_own, 4), "p_accept": round(c.p_accept, 4),
                 "p_accept_optimistic": round(c.p_accept_optimistic, 4),
                 "eig": round(c.eig, 4), "score": round(c.score, 4)}
                for c in allowed[:3]],
            "generated_by": "ScriptedNegotiator (no LLM)",
        }
        if hypothesis_id:
            memo["hypothesis_registered"] = hypothesis_id

        # Candidates evaluated in the *future*: the next round and the end of the horizon.
        # They ground both the value of waiting and the decision to walk away — judging with
        # the current round's snapshot made the agent accept anything or leave in round 1.
        horizon = self.protocol.max_rounds
        future_candidates: list[Candidate] = []
        for future_round in {min(round_idx + 1, horizon), horizon}:
            future_candidates.extend(score(future_round))

        # 1. Accept what is on the table?
        if offer_on_table is not None:
            u_offer = float(self.utility(offer_on_table))
            accept, detail = self.personality.should_accept(
                u_offer, future_candidates, self.utility)
            memo["acceptance_check"] = {k: round(v, 4) for k, v in detail.items()}
            if accept:
                memo["chosen"] = {"action": "accept"}
                memo["rationale"] = (
                    f"the offer on the table gives me {u_offer:.3f} utility, above my reserve "
                    f"({self.utility.reservation_value:.3f}) and above the expected value of "
                    f"continuing ({detail['ev_continue']:.3f})")
                memo["expects_to_learn"] = "nothing: the negotiation ends in agreement"
                return Action(round_idx, self.view.role_id, "accept"), memo

        # 2. Walk away? Same future basis: only if nothing reachable in what remains beats the
        # own reserve plus margin.
        walk, wdetail = self.personality.should_walk_away(future_candidates, self.utility)
        memo["walk_away_check"] = {k: round(v, 4) for k, v in wdetail.items()}
        if walk:
            memo["chosen"] = {"action": "walk_away"}
            memo["rationale"] = (
                f"not even the best reachable offer at the end of the horizon "
                f"({wdetail['best_reachable']:.3f}) beats my reserve plus margin "
                f"({wdetail['threshold']:.3f}): no zone of agreement under my belief")
            memo["expects_to_learn"] = "nothing: I am leaving"
            return Action(round_idx, self.view.role_id, "walk_away"), memo

        # 3. Propose. On the first offer the strategy's anchor rules, if there was one and it
        # respects this round's concession policy.
        if round_idx == 1 and self.first_offer is not None and self.my_last_offer is None:
            anchor = {self.declared.issue_id: self.first_offer}
            anchor_utility = float(self.utility(anchor))
            if anchor_utility >= target - 1e-9:
                memo["anchor_from_strategy"] = self.first_offer
                memo["chosen"] = {"action": "propose", "offers": [anchor]}
                memo["rationale"] = (
                    f"I open at {self.first_offer:.0f}, the anchor set in preparation "
                    f"(own utility {anchor_utility:.3f}); this round's concession policy asked "
                    f"for at least {target:.3f}")
                memo["expects_to_learn"] = (
                    "how they respond to a high anchor: their counter-offer bounds their reserve")
                self.my_last_offer = anchor
                return Action(round_idx, self.view.role_id, "propose", offers=[anchor]), memo
            memo["anchor_rejected"] = (
                f"the anchor {self.first_offer:.0f} gives utility {anchor_utility:.3f}, below "
                f"the concession target {target:.3f}")

        memo["chosen"] = {"action": "propose", "offers": [best.offer]}
        memo["rationale"] = (
            f"I ask {best.offer[self.declared.issue_id]:.0f}: it gives me {best.u_own:.3f} "
            f"utility with acceptance probability {best.p_accept:.3f} "
            f"({best.p_accept_optimistic:.3f} under the optimistic quantile tau="
            f"{self.personality.tau}) and information gain {best.eig:.3f} nats; the planned "
            f"concession for this round does not allow going below {target:.3f}")
        memo["expects_to_learn"] = (
            f"if they reject, the counterpart's rv is above "
            f"{self.estimate_counterpart_utility(best.offer):.3f} in its scale; if they "
            f"accept, below")
        self.my_last_offer = best.offer
        return Action(round_idx, self.view.role_id, "propose", offers=[best.offer]), memo

    # ------------------------------------------------------------------ preparation
    def adopt_strategy(self, strategy: dict) -> dict:
        """Take from the planner's strategy what is *its* decision: the aspiration and the
        opening anchor. Everything else (beliefs, probabilities) keeps coming from the foxes.

        Validated against the case's space: an aspiration or anchor out of range is ignored
        and recorded, instead of breaking the negotiation.
        """
        self.strategy = strategy
        issue = self.view.domain.issues[0]
        adopted: dict = {"aspiration_utility": None, "first_offer": None, "rejected": []}
        params = (strategy.get("strategy", {}) or {}).get("negotiation_parameters", {}) or {}

        raw_aspiration = ((strategy.get("declared_utility", {}) or {})
                          .get("aspiration", {}) or {}).get("value")
        if isinstance(raw_aspiration, (int, float)) and issue.low <= raw_aspiration <= issue.high:
            utility = float(self.utility({issue.issue_id: float(raw_aspiration)}))
            if utility > self.utility.reservation_value:
                self.aspiration = utility
                adopted["aspiration_utility"] = utility
            else:
                adopted["rejected"].append(
                    f"aspiration {raw_aspiration} is below the party's own reservation value")
        elif raw_aspiration is not None:
            adopted["rejected"].append(f"aspiration {raw_aspiration} is outside the case range")

        raw_first = (params.get("first_offer") or {}).get("value")
        if isinstance(raw_first, (int, float)) and issue.low <= raw_first <= issue.high:
            step = next((spec.get("step") for spec in self.view.issues_spec
                         if spec["id"] == issue.issue_id), None)
            value = (round((raw_first - issue.low) / step) * step + issue.low if step
                     else float(raw_first))
            self.first_offer = float(value)
            adopted["first_offer"] = self.first_offer
        elif raw_first is not None:
            adopted["rejected"].append(f"first offer {raw_first} is outside the case range")
        adopted["online_foxes"] = sorted(self.foxes)
        adopted["foxes_source"] = self.foxes_source
        return adopted

    def derive_aspiration(self, beliefs: Beliefs) -> float:
        """The aspiration comes from the belief, not from a constant.

        It is what the agent believes it can really get at the end of the horizon, evaluated
        under the personality's optimistic quantile. That way tau moves the whole concession
        policy — which is what the experiment studies — instead of being masked by a fixed number.
        """
        candidates = self.personality.score_offers(
            self.space, self.utility, self.protocol.max_rounds, beliefs.theta,
            est_counterpart_utility=self.estimate_counterpart_utility)
        reachable = [c.u_own for c in candidates if c.p_accept_optimistic >= 0.25]
        floor = self.utility.reservation_value + 0.05
        if not reachable:
            return min(max(candidates[0].u_own, floor), 0.95)
        return float(min(max(max(reachable), floor), 0.95))

    def preparation_artifacts(self) -> dict:
        """Prior distributions and outcomes, before the first round (§5.1)."""
        beliefs = self.beliefs(round_idx=0)
        # If the planner set an aspiration, it rules: it is a strategy decision, not an
        # estimate. Otherwise it is derived from the belief.
        if self.aspiration is None:
            self.aspiration = self.derive_aspiration(beliefs)
        # The value of information depends on *when*: in round 1 the counterpart asks for
        # almost everything whatever its rv, so knowing it changes nothing. EVPI appears once
        # the counterpart has conceded and its rv starts to discriminate between deals.
        horizon = self.protocol.max_rounds
        by_round: dict[str, float] = {}
        detail: dict[str, dict] = {}
        for label, r in (("1", 1), (str(horizon // 2), horizon // 2), (str(horizon), horizon)):
            info = evpi(self.space, self.utility, r, beliefs.theta,
                        est_counterpart_utility=self.estimate_counterpart_utility)
            by_round[label] = float(info["evpi"])
            detail[label] = {"best_offer_ex_ante": info["best_offer_ex_ante"],
                             "expected_utility_ex_ante": info["expected_utility_ex_ante"],
                             "expected_utility_with_perfect_info":
                                 info["expected_utility_with_perfect_info"]}
        return {"belief": beliefs.summary,
                "aspiration_utility": self.aspiration,
                "online_foxes": sorted(self.foxes),
                "foxes_source": self.foxes_source,
                "evpi_rv_by_round": by_round,
                "evpi_detail": detail,
                "declared_utility": self.declared.to_dict()}
