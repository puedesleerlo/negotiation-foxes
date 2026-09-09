---
name: f04-acceptance-boundary
description: >-
  Estimates the probability that the counterpart accepts an offer. NOT validated: within a single session there is not enough data to identify the boundary, and it performs worse than predicting the base rate. Use it only as experimental and keep it out of the pool.
---

# f04_accept_boundary_kde — Acceptance boundary (experimental)

## 1. When to use it and when not

**Do not use it to decide in E1.** It is `implemented`, not `validated`, and the registry blocks
it unless you ask for `allow_experimental=True`. Its output does not enter the pool.

The reason is a finding, not an oversight: its declared scope needs ≥ 4 responses *with
variation*, and in the alternating-offers protocol the counterpart rejects the first four offers
almost always. Without a single observed acceptance, the boundary is not identified.

To estimate acceptance probability in E1, use the pool's posterior over (rv, beta, T) and
compute `P(u_cp(offer) ≥ target_cp(t))` — `tools.decision.p_accept_from_belief` (DECISIONS
D-026). It is the same quantity, estimated where there is data.

Since the architecture review (REVIEW F5) the negotiator does feed responses to the foxes — the
implied `reject` of its own previous offer and the counterpart's `accept` — so f04 now receives
its inputs through `update()` in a program. The scope condition above still applies.

## 2. What inputs it needs and where they come from

| Input | Source |
|---|---|
| observed responses (offer → accepted/rejected) | gym trace, through `Observation.action` |
| estimated utility of each offer to the counterpart | single-issue: complementary utility; multi-issue: f02/f09 |
| population prior | `assets/prior.json`, calibrated on the synthetic dataset |

## 3. How to call it

```python
from foxes.registry import FoxRegistry

reg = FoxRegistry()
fox = reg.create("f04_accept_boundary_kde", allow_experimental=True)   # requires the flag
est = fox.p_accept(state, offer, utility_to_counterpart=0.62, t=0.4)
print(est.value, est.low, est.high)
```

Validation: `python -m foxes.validate --fox f04_accept_boundary_kde` · Calibration: `python -m foxes.calibrate --fox f04_accept_boundary_kde`

## 4. How to read the output

A `ProbEstimate` with `value`, `low` and `high`. Never use `value` without looking at the
interval: under E1's conditions it usually covers almost the whole range, which is the honest
way of saying it does not know.

## 5. Known failure modes

- **No variation in the responses, no boundary**: the dominant failure mode.
- **Systematic over-prediction**: without the population prior it predicts 0.10 where 0.03 is
  observed. With the prior, the Brier drops from 0.075 to 0.049, but climatology (0.043) still wins.
- **Moving boundary**: the counterpart lowers its threshold as its deadline approaches; the
  time term models that crudely.

## 6. Typical cost

Milliseconds. Grid of 4×41×5 = 820 combinations. No LLM.

## 7. Evidence

- `baarslag2015lear` §4.3 (kernel density estimation) and §5.1.2 (learning the acceptance
  strategy by recording which offers were accepted and which were not).

## 8. Minimal test

`pytest foxes/tests/test_foxes.py -k f04` — the estimated probability must grow with the
utility the offer gives the counterpart, the interval must contain the value, and responses
must reach the fox through `update()`.
