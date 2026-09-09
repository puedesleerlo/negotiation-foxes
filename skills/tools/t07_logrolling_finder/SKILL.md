---
name: t07-trade-finder
description: >-
  Finds trades between issues that raise the expected joint utility: yielding on what matters little to you in exchange for what matters a lot. Needs a belief about the counterpart's weights.
---

# t07_logrolling_finder — Trade finder (logrolling)

## 1. When to use it and when not
**Use it** when f02 or f09 are in scope and you have a base offer on the table.

**It does not apply** with a single issue, nor without samples of the counterpart's weights: in
both cases it says so instead of inventing a trade.

## 2. Inputs
| Input | Source |
|---|---|
| base offer | table |
| your utility | preparation |
| samples of the counterpart's weights and directions | f02 / f09 |

## 3. How to call it
```python
from tools.integrative import find_logrolling

trades = find_logrolling(domain, my_utility, base_offer, weight_samples, directions)
if trades:
    print(trades[0].give_issue, '->', trades[0].take_issue)
```

## 4. How to read the output
A list of trades ordered by expected joint gain, with the delta for each side.

## 5. Failure modes
- **Inverted directions**: if the belief about the counterpart's preference direction is
  backwards, it will propose trades that make both worse off.
- **Expected gain, not guaranteed**: computed under your belief; if it is wide, the proposed
  trade may not be a gain under the truth.

## 6. Cost
Milliseconds: it walks the space comparing against the base offer.

## 7. Evidence
- L2 of the report: Vetschera 2013 finds that a tool recommending logrolling leads to
  agreements with higher joint utility. Reference in the review queue.

## 8. Minimal test
`pytest tests/test_integrative_tools.py -k logrolling` — only proposes joint improvements,
ordered, and declares itself not applicable without issues or without a belief.
