---
name: t08-contingent-clause
description: >-
  Turns a difference in beliefs about a future event into value for both parties: instead of arguing about who is right, they bet. Without a sufficient difference in beliefs, it proposes nothing.
---

# t08_contingent_builder — Contingent-clause builder

## 1. When to use it and when not
**Use it** when you identify a future event about which you and the counterpart hold clearly
different beliefs (by default, a minimum difference of 0.10).

**Warning**: it is the tool with the least backing in the catalog. See section 7.

## 2. Inputs
| Input | Source |
|---|---|
| event and your probability | your analysis of the case |
| the counterpart's estimated probability | inferred from its messages or offers |
| size of the bet | concession policy |

## 3. How to call it
```python
from tools.integrative import build_contingent

c = build_contingent('rezoning within 2 years', 0.70, 0.30, stake=0.2)
if c: print(c.payment_if_event, c.own_expected_value, c.counterpart_expected_value)
```

## 4. How to read the output
The two payments (if the event happens and if not) and the expected value for each party
**under its own belief**. The bet is priced at the midpoint between the two beliefs, so the
surplus (stake × difference ÷ 2) is split equally (DECISIONS D-029).

## 5. Failure modes
- **Invented counterpart belief**: the counterpart's number is usually a guess. If you get its
  direction wrong, the clause harms it and it will reject it.
- **Verifiability**: a clause on an event that cannot be verified cheaply and objectively is a
  source of dispute, not of value. The tool does not check this: it is the agent's judgement.
- **Risk**: `econ` is risk-neutral. A risk-averse counterpart may reject a bet with positive
  expected value for it.

## 6. Cost
Microseconds.

## 7. Evidence
- L2 of the report, section 3, is explicit: contingent contracts **are mentioned** as a mechanism
  to create value from disagreement, but *no computational treatment or empirical
  quantification was found* in the retrieved literature. This tool implements the
  mutually-favourable-bet criterion; it replicates no published result. Treat it as a hypothesis,
  not a validated technique.

## 8. Minimal test
`pytest tests/test_integrative_tools.py -k "contingent or clause"` — positive expected value for
both parties, symmetry when beliefs swap, and nothing when the difference is small.
