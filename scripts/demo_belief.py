"""Part 1 demo: the full belief chain on an episode with known truth.

Takes an episode from the synthetic dataset, runs the catalog foxes on the observed offers,
pools their posteriors, propagates the result to the ZOPA and compares everything against the
episode's sealed truth. It is the miniature equivalent of what the gym does in Part 3, without
LLM agents.

Usage:  python scripts/demo_belief.py [--family conceder] [--domain d2_case3] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from foxes.base import Observation                       # noqa: E402
from foxes.f07_zopa_pareto_estimator.fox import utility_from_theta  # noqa: E402
from foxes.domain import Utility                          # noqa: E402
from foxes.pooling import disagreement, linear_pool, leave_one_out  # noqa: E402
from foxes.registry import FoxRegistry                    # noqa: E402
from foxes.scoring import score_posterior                 # noqa: E402
from foxes.synthetic import DOMAINS                       # noqa: E402


def build(row: pd.Series, key: str, domain) -> Utility:
    maps = {k: (tuple(v) if isinstance(v, list) else v)
            for k, v in json.loads(row[f"{key}_value_maps"]).items()}
    util = Utility(domain, dict(zip(domain.issue_ids, json.loads(row[f"{key}_weights"]))), maps)
    util.reservation_value = float(row[f"{key}_rv"])
    return util


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="conceder")
    ap.add_argument("--domain", default="d2_case3")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    episodes = pd.read_parquet("data/synthetic/episodes.parquet")
    moves = pd.read_parquet("data/synthetic/moves.parquet")
    pick = episodes[(episodes.domain_id == args.domain) & (episodes.cp_family == args.family)]
    ep = pick.sample(1, random_state=args.seed).iloc[0]
    mv = moves[moves.episode_id == ep.episode_id]
    domain = DOMAINS[args.domain]

    print(f"Episode {ep.episode_id} · domain {args.domain} · counterpart {args.family}")
    print(f"Counterpart's sealed truth: rv={ep.cp_rv:.3f} beta={ep.cp_beta:.2f} "
          f"deadline={ep.cp_deadline:.0f} weights={[round(w, 3) for w in json.loads(ep.cp_weights)]}")
    print(f"Rounds played: {ep.n_rounds} · agreement: {'yes' if ep.agreement else 'no'}\n")

    observations = [
        Observation(round=int(r["round"]), party="cp", offer=json.loads(r["offer"]),
                    action="propose", est_utility_to_proposer=float(r["u_counterpart"]),
                    utility_to_observer=float(r["u_protagonist"]), max_rounds=24)
        for _, r in mv[mv.party == "counterpart"].iterrows()
    ]
    print(f"Counterpart observations: {len(observations)}\n")

    reg = FoxRegistry()
    posts: dict[str, list] = {}
    print("== foxes ==")
    for fox_id in ["f01_bayes_rv_concession", "f03_time_concession_regression",
                   "f02_concession_issue_weights", "f09_hypothesis_issue_weights",
                   "f06_uninformed_prior"]:
        kwargs = {"params": ["rv"]} if fox_id.startswith("f06") else {}
        fox = reg.create(fox_id, **kwargs)
        state = fox.init_state(domain)
        state.data["counterpart_party"] = "cp"
        state.program_id, state.party = f"demo-{ep.episode_id}", "protagonist"
        for obs in observations:
            state = fox.update(state, obs)
        post = reg.call(fox, state, step="negotiation")
        mark = "in scope" if post.scope_ok else "OUT OF SCOPE (does not enter the pool)"
        detail = ", ".join(
            f"{p}: median {post.quantile(p, 0.5):.3f} [{post.interval(p, 0.9)[0]:.3f}, "
            f"{post.interval(p, 0.9)[1]:.3f}]" for p in post.params[:3])
        print(f"  {fox_id:32} {mark}\n      {detail}")
        for param in post.params:
            posts.setdefault(param, []).append(post)

    print("\n== equal-weight pool and comparison with the truth ==")
    truth = {"rv": float(ep.cp_rv), "beta": float(ep.cp_beta), "T": float(ep.cp_deadline)}
    weights_true = json.loads(ep.cp_weights)
    truth.update({f"w.{i}": float(w) for i, w in zip(domain.issue_ids, weights_true)})
    for param, group in posts.items():
        # The control f06 does NOT enter the pool: it is the reference the pool is measured against.
        usable = [p for p in group if p.scope_ok and p.fox_id != "f06_uninformed_prior"]
        control = next((p for p in group if p.fox_id == "f06_uninformed_prior"), None)
        if len(usable) < 1 or param not in truth:
            continue
        pooled = linear_pool(usable, param)
        score = score_posterior(pooled, param, truth[param])
        cover = "yes" if score["cover90"] else "NO"
        ctl_txt = ""
        if control is not None:
            ctl = score_posterior(control, param, truth[param])
            ctl_txt = f" · control: CRPS {ctl['crps']:.3f}"
        print(f"  {param:14} truth {truth[param]:7.3f} · pool {score['median']:7.3f} "
              f"· CRPS {score['crps']:.3f}{ctl_txt} · covered at 90 %: {cover} "
              f"· foxes in the pool: {len(usable)}")
        if len(usable) > 1:
            print(f"      disagreement between foxes: {disagreement(usable, param):.2f} pool sd"
                  f" · leave-one-out available for: {', '.join(leave_one_out(usable, param))}")

    print("\n== propagation to the ZOPA (f07) ==")
    own = build(ep, "prot", domain)
    other_true = utility_from_theta(
        domain, np.array(weights_true),
        np.array([1.0 if isinstance(v, list) and v[1] > v[0] else
                  (1.0 if not isinstance(v, list) else -1.0)
                  for v in json.loads(ep.cp_value_maps).values()]), float(ep.cp_rv))
    space = domain.outcome_space()
    true_zopa = float(np.mean([(own(o) >= own.reservation_value)
                               and (other_true(o) >= other_true.reservation_value) for o in space]))
    rng = np.random.default_rng(args.seed)
    n = 120
    rv_pool = linear_pool([p for p in posts["rv"]
                           if p.scope_ok and p.fox_id != "f06_uninformed_prior"], "rv")
    theta = {
        "weights": rng.dirichlet(np.ones(len(domain.issues)), n),
        "directions": rng.choice([-1, 1], (n, len(domain.issues))),
        "rv": rv_pool.resample(n, rng)[:, 0],
    }
    f07 = reg.create("f07_zopa_pareto_estimator", n_draws=n)
    state = f07.init_state(domain, own)
    state.data["theta_samples"] = theta
    post = reg.call(f07, state, step="negotiation")
    lo, hi = post.interval("zopa_fraction", 0.9)
    print(f"  true ZOPA: {true_zopa:.3f} · estimated: {post.mean('zopa_fraction'):.3f} "
          f"[{lo:.3f}, {hi:.3f}] · P(no ZOPA) = {f07.p_no_zopa(state):.3f}")
    print(f"  own utility at the estimated Nash point: {post.mean('u_own_at_nash'):.3f}")

    out = Path("runs/demo_part1"); out.mkdir(parents=True, exist_ok=True)
    reg.dump_calls(out / "fox_calls.jsonl")
    print(f"\n{len(reg.calls)} fox calls recorded in {out/'fox_calls.jsonl'}")


if __name__ == "__main__":
    main()
