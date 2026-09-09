"""Task 1.9 — Generation of the synthetic calibration and validation dataset.

Generates bilateral negotiation episodes with known θ, in three domains and five counterpart
families, following the experimental design of hendrikx2012eval: everything is fixed except
the counterpart family and the scenario, so the estimator's error can be attributed.

Outputs (parquet) in data/synthetic/:
  episodes.parquet  one row per episode, with the counterpart's true θ
  moves.parquet     one row per action, with the true utility of the offer to each party
  outcomes.parquet  result, utilities, distance to Pareto/Nash, rounds

Note for reading the validations: the counterpart families are Faratin–Sierra–Jennings
tactics by definition, i.e. the same functional family f03 fits. f03's in-family validation is
therefore optimistic; the out-of-family evidence comes from real programs (see REVIEW.md, F11).

Usage:  python -m foxes.synthetic --episodes 2000 --seed 20260905
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from foxes.domain import (Counterpart, Domain, Issue, Offer, Utility, kalai_smorodinsky,
                          nash_point, pareto_frontier, target_utility, theta_of)

DOMAINS: dict[str, Domain] = {
    # A single continuous issue: the shape of the Elmtree case (Raiffa, "two parties, one issue").
    "d1_price": Domain("d1_price", (Issue("price", "continuous", low=0.0, high=1.0, steps=21),)),
    # Three issues: the shape E1 needs (price + two discrete issues).
    "d2_case3": Domain("d2_case3", (
        Issue("price", "continuous", low=0.0, high=1.0, steps=11),
        Issue("closing", "discrete", values=("30d", "60d", "90d")),
        Issue("employment", "discrete", values=("none", "1yr", "2yr")),
    )),
    # Five issues: where the weight estimators are put to the test.
    "d3_five": Domain("d3_five", (
        Issue("price", "continuous", low=0.0, high=1.0, steps=9),
        Issue("closing", "discrete", values=("30d", "60d", "90d")),
        Issue("employment", "discrete", values=("none", "1yr", "2yr")),
        Issue("warranty", "discrete", values=("none", "6m", "12m")),
        Issue("training", "discrete", values=("no", "yes")),
    )),
}


def sample_utility(domain: Domain, rng: np.random.Generator, oppose: float = 0.7,
                   reference: Utility | None = None) -> Utility:
    """Random additive utility. `oppose` = probability of preferring the opposite of `reference`."""
    weights = dict(zip(domain.issue_ids, rng.dirichlet(np.ones(len(domain.issues)))))
    value_maps: dict[str, dict | tuple] = {}
    for issue in domain.issues:
        if issue.type == "continuous":
            if reference is not None and rng.random() < oppose:
                worst, best = reference.value_maps[issue.issue_id][1], reference.value_maps[issue.issue_id][0]
            else:
                worst, best = (issue.low, issue.high) if rng.random() < 0.5 else (issue.high, issue.low)
            value_maps[issue.issue_id] = (worst, best)
        else:
            scores = np.linspace(0.0, 1.0, len(issue.values))
            if reference is not None and rng.random() < oppose:
                ref = reference.value_maps[issue.issue_id]
                order = np.argsort([ref[v] for v in issue.values])[::-1]
            else:
                order = rng.permutation(len(issue.values))
            value_maps[issue.issue_id] = {v: float(scores[list(order).index(i)])
                                          for i, v in enumerate(issue.values)}
    return Utility(domain, weights, value_maps)


def sample_counterpart(domain: Domain, rng: np.random.Generator, family: str,
                       reference: Utility) -> Counterpart:
    util = sample_utility(domain, rng, reference=reference)
    util.reservation_value = float(rng.uniform(0.15, 0.55))
    if family == "boulware":
        e = float(rng.uniform(0.10, 0.60))
    elif family == "conceder":
        e = float(rng.uniform(1.20, 4.00))
    elif family == "linear":
        e = 1.0
    elif family == "hardliner":
        e = 0.01
    else:  # tit_for_tat
        e = 1.0
    return Counterpart(utility=util, family=family, e=e,
                       deadline=int(rng.integers(12, 31)), k=float(rng.uniform(0.0, 0.15)),
                       tft_alpha=float(rng.uniform(0.5, 1.5)),
                       rng=np.random.default_rng(int(rng.integers(1 << 31))))


def run_episode(domain: Domain, episode_id: str, seed: int, family: str,
                max_rounds: int = 24) -> tuple[dict, list[dict], dict]:
    rng = np.random.default_rng(seed)
    space = domain.outcome_space()
    prot_u = sample_utility(domain, rng)
    prot_u.reservation_value = float(rng.uniform(0.15, 0.50))
    cp = sample_counterpart(domain, rng, family, reference=prot_u)

    prot_e = float(rng.uniform(0.2, 3.0))
    prot_deadline = int(rng.integers(12, 31))
    prot_vals = np.array([prot_u(o) for o in space])
    cp_vals = np.array([cp.utility(o) for o in space])

    moves: list[dict] = []
    agreement, agreed_offer, rounds_used = False, None, 0
    last_cp_util_to_prot, cp_concession = None, 0.0

    for round_idx in range(1, max_rounds + 1):
        rounds_used = round_idx
        # --- protagonist's turn
        t = min(round_idx / prot_deadline, 1.0)
        target = target_utility(t, prot_e, p_min=prot_u.reservation_value, p_max=1.0)
        idx = int(np.argmin(np.abs(prot_vals - target)))
        offer = space[idx]
        accepted = cp.accepts(offer, round_idx, cp_concession)
        moves.append({"episode_id": episode_id, "round": round_idx, "party": "protagonist",
                      "action": "propose", "offer": json.dumps(offer, default=str),
                      "u_protagonist": float(prot_u(offer)), "u_counterpart": float(cp.utility(offer)),
                      "target_utility": float(target), "accepted_by_other": bool(accepted)})
        if accepted:
            agreement, agreed_offer = True, offer
            break

        # --- counterpart's turn
        cp_target = cp.target(round_idx, cp_concession)
        cp_offer = cp.choose_offer(space, cp_target, cp_vals)
        u_to_prot = float(prot_u(cp_offer))
        if last_cp_util_to_prot is not None:
            cp_concession = max(u_to_prot - last_cp_util_to_prot, 0.0)
        last_cp_util_to_prot = u_to_prot
        prot_target = target_utility(min((round_idx + 0.5) / prot_deadline, 1.0), prot_e,
                                     p_min=prot_u.reservation_value, p_max=1.0)
        prot_accepts = u_to_prot >= max(prot_u.reservation_value, prot_target - 1e-9)
        moves.append({"episode_id": episode_id, "round": round_idx, "party": "counterpart",
                      "action": "propose", "offer": json.dumps(cp_offer, default=str),
                      "u_protagonist": u_to_prot, "u_counterpart": float(cp.utility(cp_offer)),
                      "target_utility": float(cp_target), "accepted_by_other": bool(prot_accepts)})
        if prot_accepts:
            agreement, agreed_offer = True, cp_offer
            break

    theta = theta_of(cp)
    episode = {
        "episode_id": episode_id, "domain_id": domain.domain_id, "seed": seed,
        "cp_family": family, "cp_rv": theta["rv"], "cp_beta": theta["beta"],
        "cp_deadline": theta["T"], "cp_weights": json.dumps(theta["w"]),
        "cp_k": cp.k, "cp_value_maps": json.dumps(cp.utility.value_maps, default=str),
        "prot_rv": prot_u.reservation_value, "prot_beta": prot_e,
        "prot_deadline": float(prot_deadline),
        "prot_weights": json.dumps(prot_u.weight_vector().tolist()),
        "prot_value_maps": json.dumps(prot_u.value_maps, default=str),
        "n_rounds": rounds_used, "agreement": agreement,
        "agreed_offer": json.dumps(agreed_offer, default=str) if agreed_offer else None,
    }

    u_prot = float(prot_u(agreed_offer)) if agreement else prot_u.reservation_value
    u_cp = float(cp.utility(agreed_offer)) if agreement else cp.utility.reservation_value
    frontier = pareto_frontier(space, prot_u, cp.utility)
    frontier_pts = np.array([[prot_u(o), cp.utility(o)] for o in frontier])
    dist_pareto = float(np.min(np.hypot(frontier_pts[:, 0] - u_prot, frontier_pts[:, 1] - u_cp)))
    nash_offer, _ = nash_point(space, prot_u, cp.utility)
    ks_offer = kalai_smorodinsky(space, prot_u, cp.utility)
    outcome = {
        "episode_id": episode_id, "agreement": agreement, "u_protagonist": u_prot,
        "u_counterpart": u_cp, "joint_utility": u_prot + u_cp,
        "nash_product": max(u_prot - prot_u.reservation_value, 0) * max(u_cp - cp.utility.reservation_value, 0),
        "dist_to_pareto": dist_pareto,
        "dist_to_nash": float(np.hypot(prot_u(nash_offer) - u_prot, cp.utility(nash_offer) - u_cp)),
        "dist_to_ks": float(np.hypot(prot_u(ks_offer) - u_prot, cp.utility(ks_offer) - u_cp)),
        "rounds": rounds_used, "n_pareto": len(frontier),
    }
    return episode, moves, outcome


def generate(n_per_domain: int, seed: int, out_dir: Path) -> None:
    from foxes.domain import FAMILIES
    rng = np.random.default_rng(seed)
    episodes, moves, outcomes = [], [], []
    for domain_id, domain in DOMAINS.items():
        for i in range(n_per_domain):
            family = FAMILIES[i % len(FAMILIES)]
            ep_seed = int(rng.integers(1 << 31))
            episode_id = f"{domain_id}-{i:05d}"
            ep, mv, oc = run_episode(domain, episode_id, ep_seed, family)
            episodes.append(ep)
            moves.extend(mv)
            outcomes.append(oc)
        print(f"{domain_id}: {n_per_domain} episodes")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(episodes).to_parquet(out_dir / "episodes.parquet", index=False)
    pd.DataFrame(moves).to_parquet(out_dir / "moves.parquet", index=False)
    pd.DataFrame(outcomes).to_parquet(out_dir / "outcomes.parquet", index=False)
    df = pd.DataFrame(episodes)
    print(f"\n{len(episodes)} episodes, {len(moves)} moves -> {out_dir}")
    print(f"agreements: {df.agreement.mean():.1%}; median rounds: {df.n_rounds.median():.0f}")
    print(df.groupby("cp_family").agg(agreement=("agreement", "mean"),
                                      rounds=("n_rounds", "mean")).round(3).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthetic dataset (task 1.9)")
    ap.add_argument("--episodes", type=int, default=2000, help="episodes per domain")
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--out-dir", default="data/synthetic")
    args = ap.parse_args()
    generate(args.episodes, args.seed, Path(args.out_dir))


if __name__ == "__main__":
    main()
