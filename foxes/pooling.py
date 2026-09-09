"""Pooling of posteriors: equal-weight linear pool (E1's default).

Rule 3 of protocol §6.5. The L3 evidence backs the default: in forecast hubs with resolved
outcomes, the equal-weight ensemble is hard to beat and the most influential choice was using
the median instead of the mean. Both forms are implemented; equal weights stay the default, and
the individual posteriors are always kept so other combinations and the attribution can be
recomputed later (leave-one-out in feedback).
"""
from __future__ import annotations

import numpy as np

from foxes.base import Posterior

POOL_VERSION = "equal_weight_linear/0.1.0"


def linear_pool(posteriors: list[Posterior], param: str, n: int = 4000,
                seed: int = 0) -> Posterior:
    """Equal-weight linear mixture: each posterior is resampled and the draws concatenated."""
    usable = [p for p in posteriors if param in p.params and p.scope_ok]
    if not usable:
        raise ValueError(f"no in-scope fox estimates {param}")
    rng = np.random.default_rng(seed)
    per = max(n // len(usable), 1)
    chunks, sources = [], []
    for post in usable:
        idx = rng.choice(len(post.samples), size=per, p=post.weights)
        chunks.append(post.column(param)[idx])
        sources.extend([post.fox_id] * per)
    samples = np.concatenate(chunks).reshape(-1, 1)
    pooled = Posterior("pool", POOL_VERSION, [param], samples,
                       program_id=usable[0].program_id, party=usable[0].party,
                       round=usable[0].round)
    pooled.warnings = [f"pool of {len(usable)} foxes: {', '.join(p.fox_id for p in usable)}"]
    return pooled


def median_pool(posteriors: list[Posterior], param: str, n: int = 4000,
                seed: int = 0) -> Posterior:
    """Quantile-median variant (Vincentisation). Recorded, not enabled by default."""
    usable = [p for p in posteriors if param in p.params and p.scope_ok]
    if not usable:
        raise ValueError(f"no in-scope fox estimates {param}")
    grid = np.linspace(0.001, 0.999, 512)
    quantiles = np.array([[p.quantile(param, q) for q in grid] for p in usable])
    merged = np.median(quantiles, axis=0)
    rng = np.random.default_rng(seed)
    samples = np.interp(rng.uniform(0, 1, n), grid, merged).reshape(-1, 1)
    pooled = Posterior("pool_median", "median_quantile/0.1.0", [param], samples,
                       program_id=usable[0].program_id, party=usable[0].party,
                       round=usable[0].round)
    pooled.warnings = [f"median pool of {len(usable)} foxes"]
    return pooled


def disagreement(posteriors: list[Posterior], param: str) -> float:
    """Disagreement between foxes in standard deviations of the pool (triggers hypotheses, rule 4)."""
    usable = [p for p in posteriors if param in p.params and p.scope_ok]
    if len(usable) < 2:
        return 0.0
    medians = np.array([p.quantile(param, 0.5) for p in usable])
    pooled = linear_pool(usable, param)
    sd = float(np.std(pooled.column(param)))
    return float((medians.max() - medians.min()) / max(sd, 1e-9))


def leave_one_out(posteriors: list[Posterior], param: str) -> dict[str, Posterior]:
    """Pools without each fox, for the feedback's attribution (§5.3)."""
    usable = [p for p in posteriors if param in p.params and p.scope_ok]
    out: dict[str, Posterior] = {}
    for post in usable:
        rest = [p for p in usable if p is not post]
        if rest:
            out[post.fox_id] = linear_pool(rest, param)
    return out
