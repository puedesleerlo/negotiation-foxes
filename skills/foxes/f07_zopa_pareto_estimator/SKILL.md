---
name: f07-zopa-and-frontier
description: >-
  Propagates the pool's θ samples to what matters for deciding: size of the ZOPA, reachable utility at the Nash point and probability that there is no zone of agreement. Use it in preparation and before every important offer.
---

# f07_zopa_pareto_estimator — ZOPA, Pareto frontier and Nash point

## 1. When to use it and when not

**Use it** once you have θ samples of the counterpart (from the pool) and your declared utility.
It is the step that turns a belief about parameters into a belief about outcomes.

**Do not use it** with θ samples coming from out-of-scope foxes: it propagates whatever you give
it, garbage included. Filter by `scope_ok` first.

## 2. What inputs it needs and where they come from

| Input | Source |
|---|---|
| own declared utility | `declared_utility.json` (sealed by the gym) |
| θ samples: weights, directions and rv | pool of f01/f02/f03/f09 |
| outcome space | `case.yaml` |

## 3. How to call it

```python
from foxes.f07_zopa_pareto_estimator import ZopaParetoEstimator

fox = ZopaParetoEstimator(n_draws=200)
state = fox.init_state(domain, own_utility)
state.data["theta_samples"] = {"weights": W, "directions": D, "rv": RV}
post = fox.posterior(state)
p_no_zopa = fox.p_no_zopa(state)
```

Validation: `python -m foxes.validate --fox f07_zopa_pareto_estimator`

## 4. How to read the output

Four parameters: `zopa_fraction` (fraction of the space that is mutually acceptable),
`u_own_at_nash`, `u_other_at_nash` and `joint_at_nash`. Use them as distributions: the median
of `u_own_at_nash` is a reasonable aspiration; its low quantile, a risk warning.

## 5. Known failure modes

- **Garbage in, garbage out**: if the estimated preference directions are inverted, the
  estimated ZOPA can be large when it is actually empty.
- **Sampled space**: above 400 outcomes the space is sampled; in very large domains the ZOPA
  fraction carries sampling noise (measured: ≤ 0.017 error with exact θ).
- **Non-additive utility**: out of scope. The estimator assumes additivity on both sides.

## 6. Typical cost

Tens to hundreds of milliseconds depending on `n_draws` and the size of the space.

## 7. Evidence

- `banks2020adve` — Adversarial Risk Analysis: represent what is unknown about the opponent
  with subjective distributions and propagate them by Monte Carlo instead of collapsing them to
  a point.

## 8. Minimal test

`pytest foxes/tests/test_foxes.py -k f07` — with exact θ the estimated ZOPA must match the
true one (tolerance 0.02).
