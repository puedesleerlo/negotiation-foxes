---
name: f08-outcome-simulator
description: >-
  Simulates complete negotiations against counterpart families to estimate the utility distribution and the impasse probability of a candidate strategy. Experimental: its output depends on the real counterpart resembling one of the simulated families.
---

# f08_outcome_simulator — Outcome simulator (experimental)

## 1. When to use it and when not

**Use it in preparation** to compare candidate strategies with each other, not to predict the
real outcome in absolute terms. It is `implemented`: the registry requires
`allow_experimental=True` and its output does not enter the pool.

**Do not use it** to justify a reservation value or a walk-away rule: its impasse probability is
conditional on the family assumption, which is precisely what you do not know.

## 2. What inputs it needs and where they come from

| Input | Source |
|---|---|
| candidate strategy (e, deadline) | `strategy.md` / `plan.json` |
| θ samples | pool |
| counterpart families to consider | configuration; by default the four that concede |

## 3. How to call it

```python
from foxes.registry import FoxRegistry

fox = FoxRegistry().create("f08_outcome_simulator", allow_experimental=True)
state = fox.init_state(domain, own_utility)
state.data.update(theta_samples=theta, strategy={"e": 0.5, "deadline": 20})
post = fox.posterior(state)
print(post.summary()["u_own"], fox.p_impasse(state))
```

Validation: `python -m foxes.validate --fox f08_outcome_simulator`

## 4. How to read the output

`u_own`, `u_other`, `joint_utility`, `agreement` and `rounds` as distributions over
simulations. Compare strategies by difference, not by level.

## 5. Known failure modes

- **Counterpart outside the simulated families**: the realistic case. The outcome distribution
  becomes optimistic because the simulated families concede.
- **Double counting of your own model**: if the θ samples already came from assuming a family,
  simulating against that family confirms what was assumed.

## 6. Typical cost

Hundreds of milliseconds to seconds: it simulates a full negotiation per sample.

## 7. Evidence

- `hendrikx2012eval` — experimental design for comparing models: fix everything except the
  counterpart family and the scenario. Applied forward here.
- `faratin1998nego` — the simulated families.

## 8. Minimal test

`pytest foxes/tests/test_foxes.py -k f08` — with θ and a strategy given, it returns a
distribution with the five columns and an impasse probability in [0,1]; without a strategy it
is out of scope.
