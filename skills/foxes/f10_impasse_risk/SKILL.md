---
name: f10-impasse-risk
description: >-
  CANDIDATE, not implemented: it would model the impasse risk as a function of how ambitious an offer is. Do not invoke it. It documents the only quantitative evidence in the corpus about the cost side of being ambitious.
---

# f10_impasse_risk — Impasse risk (candidate)

## 1. When to use it and when not

**Do not use it.** Status `candidate`. The reason is not lack of time but lack of data: the only
quantitative characterisation available is over 26 million single-issue eBay negotiations, and
its dataset is not accessible. Recalibrating its functional form on our synthetic dataset would
change the estimated object — human impasse versus impasse of simulated agents — and would
produce a number that looks like empirical evidence without being one.

For impasse risk in E1, use f08 in experimental mode (simulated probability of no agreement)
and treat it as what it is: conditional on the family assumption.

## 2. What inputs it needs and where they come from

| Input | Source |
|---|---|
| ambition of the offer as a fraction of the anchor | own computation |
| calibration data | **not available** |

## 3. How to call it

```python
# Blocked on purpose:
# FoxRegistry().create("f10_impasse_risk")  -> ValueError: `candidate`
```

Validation: `python -m foxes.validate --fox f10_impasse_risk`

## 4. How to read the output

When it exists: impasse probability with an interval.

## 5. Known failure modes

- **Domain transfer**: the eBay curve describes human buyers in single-issue sales. Applying it
  to a multi-issue negotiation between agents is an extrapolation.

## 6. Typical cost

Not applicable.

## 7. Evidence

- `schweinsberg2022unde` — linear anchoring on the final price (R² = .997, slope 0.82) and
  non-linear impasse risk in three zones: safety, acceleration (≈ 90 %–20 % of list price) and
  saturation (< 20 %, ≈ 95 % impasses). Estimated optimal opening ≈ 80 % of list price. Salient
  values (50 %, 67 %, 75 %, 80 %) are local risk minima.
  The PDF has no open-access route: the figures come from the Kosmos report, not the original.

## 8. Minimal test

The registry must reject it: `pytest foxes/tests/test_foxes.py -k registry_blocks`.
