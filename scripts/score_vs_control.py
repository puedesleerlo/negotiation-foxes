"""The headline comparison: the pooled belief about the counterpart's reserve vs the control.

Reruns f01 and f03 on the validation split of the synthetic dataset (the same 400-episode
sample, seed 0, that produced f01's validation report), pools the in-scope posteriors with the
equal-weight linear pool the gym uses, and scores the pool against the sealed θ next to the
uninformed control f06 **on exactly the same evaluations**. Oracle condition, as in every
validation report (`foxes/validate.py`).

Writes the numbers to stdout and, with `--plot`, the figure used in the README.

Usage:  PYTHONPATH=. python scripts/score_vs_control.py [--episodes 400] [--plot docs/img/pool_vs_control.png]
        (the plot needs matplotlib, which is not a project dependency:
         `uv run --with matplotlib python scripts/score_vs_control.py --plot ...`)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from foxes.f01_bayes_rv_concession import BayesRvConcession           # noqa: E402
from foxes.f03_time_concession_regression import TimeConcessionRegression  # noqa: E402
from foxes.f06_uninformed_prior import UninformedPrior                # noqa: E402
from foxes.pooling import linear_pool                                 # noqa: E402
from foxes.scoring import score_posterior                             # noqa: E402
from foxes.synthetic import DOMAINS                                   # noqa: E402
from foxes.validate import (OBS_POINTS, _state, bucket_label,         # noqa: E402
                            counterpart_observations, load)


def evaluate(episodes: pd.DataFrame, moves: pd.DataFrame) -> pd.DataFrame:
    foxes = [BayesRvConcession(), TimeConcessionRegression(n_bootstrap=60)]
    control = UninformedPrior(params=["rv"])
    rows = []
    for _, ep in episodes.iterrows():
        domain = DOMAINS[ep.domain_id]
        obs_all = counterpart_observations(moves[moves.episode_id == ep.episode_id])
        for k in OBS_POINTS:
            obs = obs_all[:k]
            if len(obs) < 2:
                continue
            posts = [f.posterior(_state(f, domain, obs)) for f in foxes]
            in_scope = [p for p in posts if p.scope_ok and "rv" in p.params]
            if not in_scope:
                continue                     # nothing enters the pool: the gym falls back too
            truth = float(ep.cp_rv)
            pool = score_posterior(linear_pool(in_scope, "rv"), "rv", truth)
            ctl = score_posterior(control.posterior(_state(control, domain, obs)), "rv", truth)
            rows.append({"episode_id": ep.episode_id, "family": ep.cp_family,
                         "n_obs": bucket_label(k), "n_foxes": len(in_scope),
                         **{f"pool_{m}": pool[m] for m in ("crps", "log_score", "cover90")},
                         **{f"ctl_{m}": ctl[m] for m in ("crps", "log_score", "cover90")}})
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, path: Path, n_episodes: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [b for b in ("2", "4", "8", "all") if b in set(df.n_obs)]
    g = df.groupby("n_obs").mean(numeric_only=True).loc[order]
    x = range(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, metric, label, better in ((axes[0], "log_score", "log score", "higher is better"),
                                      (axes[1], "crps", "CRPS", "lower is better")):
        ax.plot(x, g[f"pool_{metric}"], "o-", color="#b5542d", lw=2, label="pooled foxes (f01 + f03)")
        ax.plot(x, g[f"ctl_{metric}"], "s--", color="#6b6b6b", lw=1.5, label="control f06 (flat prior)")
        ax.set_xticks(list(x), [f"{b} offers" if b != "all" else "all offers" for b in order])
        ax.set_title(f"{label} ({better})", fontsize=10)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Belief about the counterpart's reservation value vs. sealed truth\n"
                 f"validation split, {n_episodes} synthetic episodes, {len(df)} matched evaluations",
                 fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"figure written to {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--plot", type=Path, default=None)
    args = ap.parse_args()

    episodes, moves = load(args.episodes, split="validation")
    df = evaluate(episodes, moves)
    n_ep = df.episode_id.nunique()
    print(f"validation split · {len(episodes)} episodes sampled (seed 0) · {n_ep} with a pooled "
          f"belief · {len(df)} matched evaluations · oracle condition\n")
    cols = ["pool_log_score", "ctl_log_score", "pool_crps", "ctl_crps", "pool_cover90", "ctl_cover90"]
    table = df.groupby("n_obs")[cols].mean().reindex(["2", "4", "8", "all"]).dropna()
    table.loc["pooled"] = df[cols].mean()
    table.insert(0, "n", df.groupby("n_obs").size().reindex(table.index).fillna(len(df)).astype(int))
    print(table.round(3).to_markdown())
    by_family = df.groupby("family")[["pool_log_score", "ctl_log_score", "pool_crps", "ctl_crps"]].mean()
    print("\nby counterpart family\n" + by_family.round(3).to_markdown())
    if args.plot:
        plot(df, args.plot, n_ep)


if __name__ == "__main__":
    main()
