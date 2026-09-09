"""Batches of programs and sweeps of tau and lambda (task 3.5).

Runs N programs varying the seed and the personality parameters, and aggregates the result into
the frontier RQ2 cares about: how much utility is gained and how much impasse risk rises as the
agent becomes more optimistic (tau) and more exploratory (lambda).

Usage:  python -m gym.batch --seeds 5 --sweep --focal parkers
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from feedback.metrics import calibration_metrics, outcome_metrics
from gym.case import CaseBundle
from gym.run import run_program

DEFAULT_TAUS = (0.50, 0.65, 0.75, 0.90)
DEFAULT_LAMBDAS = (0.0, 0.25, 1.0)


def run_batch(case_id: str = "parker_gibson", seeds: int = 5,
              taus: tuple[float, ...] = DEFAULT_TAUS,
              lambdas: tuple[float, ...] = DEFAULT_LAMBDAS,
              runs_dir: Path = Path("runs"), verbose: bool = True,
              focal_role: str | None = None, baseline_tau: float = 0.5,
              baseline_lambda: float = 0.0, max_rounds: int | None = None,
              reuse_preparation_from: str | None = None) -> pd.DataFrame:
    """With `focal_role`, tau and lambda are swept **only** in that party; the other stays fixed.

    It is the only way to read the effect of the lever: if both parties turn optimistic at the
    same time, what is measured is the interaction, not the lever.
    """
    bundle = CaseBundle.load(case_id)
    roles = bundle.role_ids()
    issue_id = bundle.domain().issue_ids[0]
    rows: list[dict] = []
    combos = list(itertools.product(taus, lambdas, range(seeds)))
    for i, (tau, lam, seed) in enumerate(combos, 1):
        if focal_role:
            other = bundle.counterpart_of(focal_role)
            tau_arg = {focal_role: tau, other: baseline_tau}
            lam_arg = {focal_role: lam, other: baseline_lambda}
        else:
            tau_arg, lam_arg = tau, lam
        result = run_program(case_id, seed=seed, tau=tau_arg, lam=lam_arg, runs_dir=runs_dir,
                             max_rounds=max_rounds, verbose=False,
                             reuse_preparation_from=reuse_preparation_from)
        metrics = outcome_metrics(bundle, result["outcome"])
        calib = calibration_metrics(bundle, runs_dir / result["program_id"] / "beliefs.parquet")
        row = {"program_id": result["program_id"], "tau": tau, "lambda": lam, "seed": seed,
               "agreement": metrics.agreement, "rounds": metrics.rounds,
               "ended_by": metrics.ended_by,
               "price": (metrics.agreed_offer[issue_id] if metrics.agreed_offer else None),
               "joint_utility": metrics.joint_utility,
               "dist_to_pareto": metrics.dist_to_pareto,
               "dist_to_nash": metrics.dist_to_nash}
        for role in roles:
            row[f"u_{role}"] = metrics.utilities[role]
            row[f"surplus_{role}"] = metrics.surplus_split[role]
        if not calib.empty:
            row["crps_mean"] = float(calib.crps.mean())
            row["cover90_mean"] = float(calib.cover90.mean())
        rows.append(row)
        if verbose and i % max(len(combos) // 10, 1) == 0:
            print(f"  {i}/{len(combos)} programs")
    return pd.DataFrame(rows)


def frontier(df: pd.DataFrame, roles: list[str], n_boot: int = 2000,
             seed: int = 0) -> pd.DataFrame:
    """Utility–impasse frontier per (tau, lambda), with bootstrap intervals."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for (tau, lam), group in df.groupby(["tau", "lambda"]):
        utilities = group[f"u_{roles[0]}"].to_numpy()
        impasse = (~group["agreement"]).astype(float).to_numpy()
        boots_u, boots_i = [], []
        for _ in range(n_boot):
            idx = rng.integers(0, len(group), len(group))
            boots_u.append(utilities[idx].mean())
            boots_i.append(impasse[idx].mean())
        rows.append({
            "tau": tau, "lambda": lam, "n": len(group),
            f"u_{roles[0]}_mean": float(utilities.mean()),
            f"u_{roles[0]}_lo": float(np.quantile(boots_u, 0.05)),
            f"u_{roles[0]}_hi": float(np.quantile(boots_u, 0.95)),
            "impasse_rate": float(impasse.mean()),
            "impasse_lo": float(np.quantile(boots_i, 0.05)),
            "impasse_hi": float(np.quantile(boots_i, 0.95)),
            "rounds_mean": float(group["rounds"].mean()),
            "price_mean": float(group["price"].dropna().mean()) if group["price"].notna().any()
            else float("nan"),
        })
    return pd.DataFrame(rows).sort_values(["tau", "lambda"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch of programs and sweep (task 3.5)")
    ap.add_argument("--case", default="parker_gibson")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--sweep", action="store_true", help="sweep tau and lambda")
    ap.add_argument("--focal", default=None,
                    help="role whose tau and lambda are swept; the other stays at the baseline")
    ap.add_argument("--out", default="runs/batch")
    ap.add_argument("--max-rounds", type=int, default=None,
                    help="shorten the horizon: with fewer rounds the impasse arm appears")
    ap.add_argument("--reuse-preparation", default=None, metavar="PROGRAM_ID",
                    help="reuse another program's strategies (keeps the strategy fixed across the sweep)")
    args = ap.parse_args()

    taus = DEFAULT_TAUS if args.sweep else (0.75,)
    lambdas = DEFAULT_LAMBDAS if args.sweep else (0.25,)
    df = run_batch(args.case, args.seeds, taus, lambdas, focal_role=args.focal,
                   max_rounds=args.max_rounds, reuse_preparation_from=args.reuse_preparation)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "programs.csv", index=False)

    roles = CaseBundle.load(args.case).role_ids()
    if args.focal:
        roles = [args.focal] + [r for r in roles if r != args.focal]
    front = frontier(df, roles)
    front.to_csv(out / "frontier.csv", index=False)
    print(f"\n{len(df)} programs · agreements {df.agreement.mean():.0%} · "
          f"mean price {df.price.dropna().mean():,.0f}")
    print(f"\nUtility–impasse frontier (party '{roles[0]}'):\n")
    print(front.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n-> {out}/programs.csv and {out}/frontier.csv")


if __name__ == "__main__":
    main()
