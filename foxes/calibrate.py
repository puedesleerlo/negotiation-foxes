"""Task 1.10 (a) — Calibration of the foxes on the calibration split of the synthetic set.

A fox, by this project's definition, is a model *calibrated on an explicit dataset*. Here the
dispersion parameters that control the width of its posteriors are fitted so that interval
coverage is nominal. The calibration split (40 % of the episodes, by hash of the episode_id) is
disjoint from the validation split: no number in a validation report comes from data seen here.

Usage:  python -m foxes.calibrate --fox f02_concession_issue_weights --episodes 200
        python -m foxes.calibrate --all
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from foxes.validate import load, validate_accept, validate_scalar, validate_weights

SKILLS = Path("skills/foxes")


def _write(fox_id: str, name: str, payload: dict) -> Path:
    target = SKILLS / fox_id / "assets" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "calibrated_on": "data/synthetic (calibration split)",
               "calibrated_at": date.today().isoformat()}
    target.write_text(json.dumps(payload, indent=2))
    print(f"-> {target}")
    return target


def calibrate_weights(fox_id: str, episodes_n: int) -> None:
    """Search the dispersion parameter whose 90 % coverage is nominal."""
    from foxes.f02_concession_issue_weights import ConcessionIssueWeights
    from foxes.f09_hypothesis_issue_weights import HypothesisIssueWeights

    episodes, moves = load(episodes_n, split="calibration")
    episodes = episodes[episodes.domain_id != "d1_price"]
    grid = ([2.0, 4.0, 8.0, 12.0, 20.0, 40.0] if fox_id.startswith("f02")
            else [0.03, 0.06, 0.10, 0.18, 0.30, 0.50])
    results = []
    for value in grid:
        fox = (ConcessionIssueWeights(concentration=value) if fox_id.startswith("f02")
               else HypothesisIssueWeights(sigma=value))
        df = validate_weights(fox, episodes, moves)
        ok = df[df.scope_ok]
        if ok.empty:
            continue
        results.append({"value": value, "cover90": float(ok.cover90.mean()),
                        "cover50": float(ok.cover50.mean()), "crps": float(ok.crps.mean()),
                        "log_score": float(ok.log_score.mean())})
        print(f"  {value:>6}: coverage90={results[-1]['cover90']:.2f} "
              f"crps={results[-1]['crps']:.3f} log_score={results[-1]['log_score']:.2f}")
    if not results:
        print("no calibration data")
        return
    best = min(results, key=lambda r: abs(r["cover90"] - 0.90))
    key = "concentration" if fox_id.startswith("f02") else "sigma"
    _write(fox_id, "calibration.json", {key: best["value"], "search": results,
                                        "criterion": "90 % coverage closest to nominal"})


def calibrate_accept(episodes_n: int) -> None:
    """Population prior over the grid (slope, threshold, time coefficient)."""
    from foxes.f04_accept_boundary_kde import AcceptBoundaryKde

    episodes, moves = load(episodes_n, split="calibration")
    fox = AcceptBoundaryKde(prior=np.zeros(1))  # no prior while calibrating
    fox.prior = None
    grid = fox._grid()
    log_prior = np.zeros(len(grid))
    n_used = 0
    for _, ep in episodes.iterrows():
        mv = moves[(moves.episode_id == ep.episode_id) & (moves.party == "protagonist")]
        if len(mv) < 4:
            continue
        state = fox.init_state(None)  # type: ignore[arg-type]
        for row in mv.itertuples():
            fox.observe_response(state, float(row.u_counterpart), bool(row.accepted_by_other),
                                 float(row.round) / 24.0)
        post = fox.posterior(state)
        log_prior += post.weights          # accumulate the per-episode posteriors
        n_used += 1
    if n_used == 0:
        print("no calibration episodes")
        return
    dens = log_prior / n_used
    dens = np.clip(dens, 1e-9, None)
    log_prior = np.log(dens / dens.sum())
    log_prior -= log_prior.max()
    _write("f04_accept_boundary_kde", "prior.json",
           {"log_prior": log_prior.tolist(), "n_episodes": n_used,
            "criterion": "mixture of the per-episode posteriors of the calibration split"})


def calibrate_time_regression(episodes_n: int) -> None:
    """Search the observation noise whose 90 % coverage of `beta` is nominal.

    Calibrated on `beta` because it is the parameter f03 claims to estimate better than the
    control; rv and T are reported but not validated (see the catalog)."""
    from foxes.f03_time_concession_regression import TimeConcessionRegression

    episodes, moves = load(episodes_n, split="calibration")
    results = []
    for noise in (0.02, 0.04, 0.06, 0.10, 0.16, 0.25):
        fox = TimeConcessionRegression(noise=noise)
        df = validate_scalar(fox, "beta", "cp_beta", episodes, moves)
        ok = df[df.scope_ok]
        if ok.empty:
            continue
        results.append({"value": noise, "cover90": float(ok.cover90.mean()),
                        "cover50": float(ok.cover50.mean()), "crps_beta": float(ok.crps.mean())})
        print(f"  noise {noise:>5}: coverage90(beta)={results[-1]['cover90']:.2f} "
              f"crps={results[-1]['crps_beta']:.3f}")
    if not results:
        print("no calibration data")
        return
    best = min(results, key=lambda r: abs(r["cover90"] - 0.90))
    _write("f03_time_concession_regression", "calibration.json",
           {"noise": best["value"], "search": results,
            "criterion": "90 % coverage of beta closest to nominal"})


def main() -> None:
    ap = argparse.ArgumentParser(description="Fox calibration (task 1.10)")
    ap.add_argument("--fox", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--episodes", type=int, default=200)
    args = ap.parse_args()
    targets = ([args.fox] if args.fox else
               ["f02_concession_issue_weights", "f09_hypothesis_issue_weights",
                "f04_accept_boundary_kde", "f03_time_concession_regression"])
    for fox_id in targets:
        print(f"\n=== calibrating {fox_id} ===")
        if fox_id.startswith(("f02", "f09")):
            calibrate_weights(fox_id, args.episodes)
        elif fox_id.startswith("f04"):
            calibrate_accept(args.episodes)
        elif fox_id.startswith("f03"):
            calibrate_time_regression(args.episodes)


if __name__ == "__main__":
    main()
