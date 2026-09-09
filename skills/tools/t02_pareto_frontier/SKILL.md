---
name: t02-pareto-frontier
description: >-
  Computes the Pareto frontier, the Nash point and the Kalai–Smorodinsky point given two utility
  functions. Deterministic. Use it with a θ sample to see which agreements leave value on the table.
---

# t02 — Pareto frontier and reference points

## 1. When to use it and when not
When you already have a counterpart utility (a θ sample, not a point estimate). With a single
sample you get *one* frontier; for the uncertainty use f07, which repeats this over many samples.

## 2. Inputs
The domain's outcome space, your utility, the counterpart's utility for that sample.

## 3. How to call it
```python
from foxes.domain import pareto_frontier, nash_point, kalai_smorodinsky
frontier = pareto_frontier(domain.outcome_space(), my_utility, counterpart_utility)
nash, product = nash_point(domain.outcome_space(), my_utility, counterpart_utility)
ks = kalai_smorodinsky(domain.outcome_space(), my_utility, counterpart_utility)
```

## 4. How to read the output
The frontier is the set of non-dominated offers. The distance of an agreement to that frontier
is the "value left on the table" the feedback reports. Nash maximises the product of surpluses
over the reservation values; Kalai–Smorodinsky equalises the gains relative to the aspiration.

## 5. Failure modes
- Large spaces: the frontier is computed by pairwise comparison over the enumerated space.
- Mis-declared reservation values shift Nash and KS even though the frontier does not change.

## 6. Cost
Milliseconds for spaces of hundreds of outcomes.

## 7. Evidence
`baarslag2015lear` §6.2 uses distance to Pareto and to Nash/KS as standard performance measures.

## 8. Minimal test
`pytest foxes/tests/test_domain.py -k "pareto or nash"` — no offer on the frontier is
dominated, and Nash and KS lie on the frontier.
