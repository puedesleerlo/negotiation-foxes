---
name: t03-zopa
description: >-
  Computes the zone of possible agreement given two utilities and two reservation values.
  Deterministic. Use it to know whether an agreement is possible before discussing how to split it.
---

# t03 — ZOPA

## 1. When to use it and when not
With a θ sample. For the *probability* that there is no ZOPA — which is what matters for
deciding whether to walk away — use f07 (`p_no_zopa`), which propagates all the uncertainty.

## 2. Inputs
Outcome space, both utilities, both reservation values.

## 3. How to call it
```python
from foxes.domain import zopa
agreements = zopa(domain.outcome_space(), my_utility, counterpart_utility)
```

## 4. How to read the output
The list of offers both parties would accept. Empty means that, *under that θ sample*, no deal
is possible: not that there is none.

## 5. Failure modes
It depends entirely on the estimated reservation value of the counterpart, which is the most
uncertain thing there is. A ZOPA computed with the median rv is a much stronger claim than the
evidence supports.

## 6. Cost
Microseconds to milliseconds.

## 7. Evidence
Standard concept; Raiffa's Elmtree case uses it explicitly ("the midpoint falls inside the zone
of potential agreement only if it lies between the two true reservation values").

## 8. Minimal test
`pytest foxes/tests/test_domain.py -k nash_and_ks` — with opposed utilities and compatible
reservation values, the ZOPA is not empty.
