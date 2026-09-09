"""Proper scoring rules and calibration diagnostics.

The corpus (L1, section 2) documents that the field evaluates opponent models with point error
(RMSE on the RV, utility distance) and almost never with proper scoring rules; the exceptions
are BOND (per-turn Brier) and FortUne Dial (Brier + calibration). E1 uses proper rules from the
start: log score, CRPS and interval coverage.
"""
from __future__ import annotations

import numpy as np

from foxes.base import Posterior


def crps_samples(samples: np.ndarray, weights: np.ndarray, truth: float) -> float:
    """CRPS in energy form: E|X - y| - 0.5 E|X - X'| (weighted samples)."""
    x = np.asarray(samples, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    w = w / w.sum()
    term1 = float(np.sum(w * np.abs(x - truth)))
    order = np.argsort(x)
    xs, ws = x[order], w[order]
    cw = np.cumsum(ws)
    # E|X - X'| = 2 * integral F(1-F); exact discrete form for weighted samples
    term2 = 2.0 * float(np.sum(np.diff(xs) * (cw[:-1] * (1.0 - cw[:-1]))))
    return term1 - 0.5 * term2


def log_score_kde(samples: np.ndarray, weights: np.ndarray, truth: float,
                  floor: float = 1e-6) -> float:
    """Log density of the truth under a Gaussian KDE of the posterior (local proper rule)."""
    x = np.asarray(samples, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    w = w / w.sum()
    mean = float(np.sum(w * x))
    var = float(np.sum(w * (x - mean) ** 2))
    n_eff = 1.0 / float(np.sum(w ** 2))
    sd = max(np.sqrt(var), 1e-4)
    bandwidth = max(1.06 * sd * n_eff ** (-1 / 5), 1e-3)  # Silverman
    dens = np.sum(w * np.exp(-0.5 * ((truth - x) / bandwidth) ** 2)) / (bandwidth * np.sqrt(2 * np.pi))
    return float(np.log(max(dens, floor)))


def covered(post: Posterior, param: str, truth: float, level: float) -> bool:
    lo, hi = post.interval(param, level)
    return bool(lo <= truth <= hi)


def score_posterior(post: Posterior, param: str, truth: float) -> dict:
    x = post.column(param)
    w = post.weights
    median = post.quantile(param, 0.5)
    return {
        "truth": float(truth),
        "median": float(median),
        "mean": post.mean(param),
        "abs_error": float(abs(median - truth)),
        "crps": crps_samples(x, w, truth),
        "log_score": log_score_kde(x, w, truth),
        "cover50": covered(post, param, truth, 0.5),
        "cover90": covered(post, param, truth, 0.9),
        "entropy": post.entropy(param),
        "width90": float(np.diff(post.interval(param, 0.9))[0]),
    }


def brier(prob: np.ndarray, outcome: np.ndarray) -> float:
    return float(np.mean((np.asarray(prob, float) - np.asarray(outcome, float)) ** 2))


def reliability_table(prob: np.ndarray, outcome: np.ndarray, bins: int = 5) -> list[dict]:
    """Reliability table: mean predicted probability vs observed frequency per bin."""
    prob = np.asarray(prob, float)
    outcome = np.asarray(outcome, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1.0 else prob <= hi)
        if not mask.any():
            rows.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": 0,
                         "pred_mean": None, "obs_freq": None})
            continue
        rows.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": int(mask.sum()),
                     "pred_mean": float(prob[mask].mean()),
                     "obs_freq": float(outcome[mask].mean())})
    return rows
