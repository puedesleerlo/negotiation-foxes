---
name: t01-own-utility
description: >-
  Evaluates how much an offer is worth to you with the additive utility function you declared.
  Deterministic; estimates nothing about the counterpart. Use it before scoring any offer.
---

# t01 — Own utility

## 1. When to use it and when not
Whenever you need the value of an offer *to your party*. Do not use it to estimate the value an
offer has for the counterpart: for that you need its weights, which f02 and f09 estimate (or, in
a single-issue case, the gym's complementary utility model).

## 2. Inputs
`declared_utility.json` (weights per issue, value functions, reservation value) and the offer.
The gym **seals** that file at the end of preparation: from then on it does not change.

## 3. How to call it
```python
from foxes.domain import Utility
u = Utility(domain, weights, value_maps, reservation_value=0.30)
u({"price": 0.62, "closing": "60d", "employment": "1yr"})
```

## 4. How to read the output
A number in [0, 1]: 0 is your worst possible outcome in the declared space, 1 the best. Always
compare it against your reservation value, not against 0.

## 5. Failure modes
- Weights that do not sum to 1 are normalised on construction; if that surprises you, your
  declaration was wrong.
- Additivity: if your real valuation has interactions between issues, this function does not
  capture them and every piece of machinery that uses it inherits that error.

## 6. Cost
Microseconds.

## 7. Evidence
Standard additive utility of the field; `baarslag2015lear` §3.3 assumes it in most estimators.

## 8. Minimal test
`pytest foxes/tests/test_domain.py -k utility` — the best offer is worth 1, the worst 0, and the
weights end up normalised.
