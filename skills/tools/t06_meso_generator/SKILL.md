---
name: t06-multiple-equivalent-offers
description: >-
  Generates k packages that are worth the same to you and are as different from each other as possible. The counterpart's choice reveals what it cares about. Requires at least two issues: with one it will tell you and produce nothing.
---

# t06_meso_generator — Multiple equivalent simultaneous offers (MESO)

## 1. When to use it and when not
**Use it** when two foxes disagree about the counterpart's weights: it is the probe that
discriminates between them without giving up utility (`probe_policy: meso_when_disagreement`).

**It does not apply** to single-issue cases. In `parker_gibson` it returns `NotApplicable`, and
that is correct: the case itself forbids introducing other issues or options.

## 2. Inputs
| Input | Source |
|---|---|
| domain and offer space | `case.yaml` |
| your declared utility | preparation |
| target utility level | concession policy |

## 3. How to call it
```python
from tools.integrative import generate_mesos

meso = generate_mesos(domain, my_utility, target_utility=0.55, k=3)
if meso:
    for o in meso.offers: print(o)
else:
    print(meso.reason)
```

## 4. How to read the output
`offers` (k packages), `spread` (dispersion in the issue space: the more, the more informative)
and `detail.max_utility_gap` (how equivalent they really are).

## 5. Failure modes
- **Approximate equivalence**: with discrete spaces there are rarely k offers of identical
  utility; check `max_utility_gap` before presenting them as equivalent.
- **Low spread**: if the packages look alike, the response discriminates nothing.

## 6. Cost
Milliseconds. Greedy selection by maximum spread over the target utility band.

## 7. Evidence
- L2 of the report: Park et al. 2019 find that a MESO-based agent increases integrative
  agreements and counterpart satisfaction compared with single offers; Tomlinson and Lewicki 2015
  recommend packages of equivalent own value to diagnose priorities without revealing information.
  **Both references are in the review queue**: the Kosmos report names them but its reference
  list is cut off before reaching them.

## 8. Minimal test
`pytest tests/test_integrative_tools.py -k meso` — equivalent for oneself, different from each
other, and not applicable with a single issue.
