---
name: t05-value-of-perfect-information
description: >-
  How much you would gain if you knew θ before offering. It is the upper bound of what any probe can be worth: if the EVPI is small, do not spend rounds learning.
---

# t05_evpi — Expected value of perfect information

## 1. When to use it and when not
**Use it in preparation** to decide whether exploring is worth it, and in feedback for the
ex-post analysis. If the EVPI over `rv` is close to zero, the problem is not that you do not
know: it is that knowing would not change your best offer.

## 2. Inputs
| Input | Source |
|---|---|
| offer space | `case.yaml` |
| your declared utility | preparation |
| θ samples | pool |

## 3. How to call it
```python
from tools.decision import evpi

res = evpi(space, my_utility, round_, theta, weights)
print(res["evpi"], res["best_offer_ex_ante"])
```

## 4. How to read the output
`evpi` in units of your utility, plus the best offer without information and the expected
utility with and without perfect information.

## 5. Failure modes
- **It is a bound, not a promise**: no real probe delivers perfect information.
- **It depends on the acceptance model**: it uses the same functional form as the rest (a target
  descending towards the rv). If the counterpart does not behave that way, the EVPI measures the
  value of information inside a wrong model.

## 6. Cost
Milliseconds to tens: an offers × samples matrix.

## 7. Evidence
- `banks2020adve` — Monte Carlo propagation of the uncertainty about the opponent through to
  the decision. L1 of the report notes that **no work** estimates the value of perfect
  information about the opponent's reservation value; this makes it explicit.

## 8. Minimal test
`pytest tests/test_decision_tools.py -k evpi` — always non-negative, and exactly zero when the
belief has no uncertainty.
