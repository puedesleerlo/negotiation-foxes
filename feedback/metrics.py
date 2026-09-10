"""Feedback: outcome and calibration metrics against the sealed truth (task 3.3).

No LLM as judge (§3): everything is computed against the case's truth or against realised
events. The prose debrief is written *from* these numbers, never the other way round.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from foxes.domain import kalai_smorodinsky, nash_point, pareto_frontier, zopa
from foxes.scoring import crps_samples, log_score_kde
from gym.case import CaseBundle
from hypotheses.thresholds import threshold_from_prose


@dataclass
class OutcomeMetrics:
    agreement: bool
    agreed_offer: Optional[dict]
    rounds: int
    ended_by: str
    utilities: dict[str, float] = field(default_factory=dict)
    joint_utility: float = 0.0
    nash_product: float = 0.0
    dist_to_pareto: float = 0.0
    dist_to_nash: float = 0.0
    dist_to_ks: float = 0.0
    surplus_split: dict[str, float] = field(default_factory=dict)
    true_zopa: tuple[float, float] = (0.0, 0.0)
    reference: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


def outcome_metrics(bundle: CaseBundle, outcome: dict) -> OutcomeMetrics:
    roles = bundle.role_ids()
    utils = {rid: bundle.true_utility(rid) for rid in roles}
    space = bundle.domain().outcome_space()
    a, b = roles
    frontier = pareto_frontier(space, utils[a], utils[b])
    nash_offer, _ = nash_point(space, utils[a], utils[b])
    ks_offer = kalai_smorodinsky(space, utils[a], utils[b])
    feasible = zopa(space, utils[a], utils[b])

    offer = outcome.get("agreed_offer")
    if outcome.get("agreement") and offer:
        realized = {rid: float(utils[rid](offer)) for rid in roles}
    else:
        realized = {rid: float(utils[rid].reservation_value) for rid in roles}

    pts = np.array([[utils[a](o), utils[b](o)] for o in frontier])
    point = np.array([realized[a], realized[b]])
    dist_pareto = float(np.min(np.hypot(pts[:, 0] - point[0], pts[:, 1] - point[1])))
    nash_pt = np.array([utils[a](nash_offer), utils[b](nash_offer)])
    ks_pt = np.array([utils[a](ks_offer), utils[b](ks_offer)])

    surplus = {rid: max(realized[rid] - utils[rid].reservation_value, 0.0) for rid in roles}
    total_surplus = sum(surplus.values())
    truth = bundle.sealed_truth()

    return OutcomeMetrics(
        agreement=bool(outcome.get("agreement")), agreed_offer=offer,
        rounds=int(outcome.get("rounds", 0)), ended_by=str(outcome.get("ended_by", "")),
        utilities=realized, joint_utility=float(sum(realized.values())),
        nash_product=float(np.prod([surplus[rid] for rid in roles])),
        dist_to_pareto=dist_pareto,
        dist_to_nash=float(np.linalg.norm(point - nash_pt)),
        dist_to_ks=float(np.linalg.norm(point - ks_pt)),
        surplus_split={rid: (surplus[rid] / total_surplus if total_surplus > 0 else 0.0)
                       for rid in roles},
        true_zopa=(truth.reservation_value(roles[0]), truth.reservation_value(roles[1])),
        reference={"nash_offer": nash_offer, "ks_offer": ks_offer,
                   "n_pareto": len(frontier), "n_zopa": len(feasible),
                   "zopa_fraction": len(feasible) / max(len(space), 1)},
    )


def calibration_metrics(bundle: CaseBundle, beliefs_path: Path) -> pd.DataFrame:
    """Score each party's belief about the other's rv, round by round.

    The truth is the counterpart's reservation value *in the counterpart's utility scale*,
    which is what the foxes estimate. The control's score is reported alongside: a belief that
    does not beat a flat prior carries no information, however narrow it looks.
    """
    if not beliefs_path.exists():
        return pd.DataFrame()
    truth = bundle.sealed_truth()
    rows: list[dict] = []
    df = pd.read_parquet(beliefs_path)
    for _, row in df.iterrows():
        other = bundle.counterpart_of(row["party"])
        true_rv = truth.reservation_value_utility(other)
        samples = np.asarray(row["quantile_samples"], dtype=float)
        weights = np.ones_like(samples)
        lo50, hi50 = np.quantile(samples, [0.25, 0.75])
        lo90, hi90 = np.quantile(samples, [0.05, 0.95])
        entry = {
            "party": row["party"], "round": int(row["round"]), "true_rv": true_rv,
            "median": float(np.median(samples)),
            "abs_error": float(abs(np.median(samples) - true_rv)),
            "crps": crps_samples(samples, weights, true_rv),
            "log_score": log_score_kde(samples, weights, true_rv),
            "cover50": bool(lo50 <= true_rv <= hi50),
            "cover90": bool(lo90 <= true_rv <= hi90),
            "sd": float(np.std(samples)),
            "foxes_in_scope": row["foxes_in_scope"],
            "disagreement_sd": float(row["disagreement_sd"]),
        }
        control_raw = row.get("control_quantiles")
        control = np.asarray(json.loads(control_raw), dtype=float) if control_raw else np.array([])
        if control.size:
            entry["control_crps"] = crps_samples(control, np.ones_like(control), true_rv)
            entry["control_log_score"] = log_score_kde(control, np.ones_like(control), true_rv)
        rows.append(entry)
    return pd.DataFrame(rows)


def attribution_leave_one_out(bundle: CaseBundle, beliefs_path: Path) -> pd.DataFrame:
    """Per-fox attribution: how much the pooled belief worsens when each fox is removed (§5.3).

    Computed over the pool that actually decided — the in-scope, validated foxes — never over
    the control, which is the reference, not a member.
    """
    if not beliefs_path.exists():
        return pd.DataFrame()
    truth = bundle.sealed_truth()
    rows: list[dict] = []
    df = pd.read_parquet(beliefs_path)
    for _, row in df.iterrows():
        per_fox = json.loads(row["per_fox_quantiles"]) if row.get("per_fox_quantiles") else {}
        if len(per_fox) < 2:
            continue
        other = bundle.counterpart_of(row["party"])
        true_rv = truth.reservation_value_utility(other)
        full = np.concatenate([np.asarray(v, dtype=float) for v in per_fox.values()])
        full_score = log_score_kde(full, np.ones_like(full), true_rv)
        for fox_id in per_fox:
            rest = np.concatenate([np.asarray(v, dtype=float)
                                   for k, v in per_fox.items() if k != fox_id])
            without = log_score_kde(rest, np.ones_like(rest), true_rv)
            rows.append({"party": row["party"], "round": int(row["round"]), "fox_id": fox_id,
                         "log_score_full": full_score, "log_score_without": without,
                         "delta_log_score": full_score - without})
    return pd.DataFrame(rows)


def resolve_hypotheses(bundle: CaseBundle, program_id: str,
                       runs_dir: Path = Path("runs")) -> dict:
    """Resolve the pending hypotheses against the sealed truth and score Brier (§5.3).

    A hypothesis about a parameter is resolved if it carries a structured `threshold` (what the
    planner is asked to produce) or if its statement yields one unambiguously. Behavioural ones,
    and those that meet neither condition, stay `unresolved` with the reason: no criterion is
    invented in order to score.
    """
    from hypotheses.store import HypothesisStore

    truth = bundle.sealed_truth()
    summary: dict[str, Any] = {}
    for role_id in bundle.role_ids():
        store = HypothesisStore(program_id, role_id, runs_dir=runs_dir)
        other = bundle.counterpart_of(role_id)
        true_rv_raw = truth.reservation_value(other)          # in issue units (USD)
        true_rv_utility = truth.reservation_value_utility(other)
        for item in store.pending():
            if item.variable != "theta.rv":
                store.mark_unresolved(
                    item.hypothesis_id,
                    f"the sealed truth of this case does not fix `{item.variable}`: only the "
                    f"reservation value is verifiable here")
                continue

            comparison = value = unit = None
            if item.threshold is not None:
                comparison = item.threshold.comparison
                value = item.threshold.values[0]
                unit = item.threshold.unit
                holds = item.threshold.holds(
                    true_rv_raw if unit.upper() == "USD" else true_rv_utility)
            else:
                parsed = threshold_from_prose(item.statement)
                if parsed is None and item.interval:
                    comparison, value, unit = "lte", item.interval[1], "utility"
                    holds = true_rv_utility <= item.interval[1]
                elif parsed is None:
                    store.mark_unresolved(
                        item.hypothesis_id,
                        "no structured `threshold` and no bound extractable from the "
                        "statement: cannot be scored without inventing a criterion")
                    continue
                else:
                    comparison, value = parsed
                    unit = "USD"
                    holds = (true_rv_raw >= value if comparison == "gte"
                             else true_rv_raw <= value)
            store.resolve(
                item.hypothesis_id, truth_value=bool(holds),
                evidence=(f"true reservation value of {other}: {true_rv_raw:,.0f} USD "
                          f"({true_rv_utility:.4f} in utility). The hypothesis claimed "
                          f"{comparison} {value:,.0f} {unit}"))
        store.save()
        summary[role_id] = store.scores()
    return summary
