---
name: t04-information-gain
description: >-
  Computes how much the response to a specific offer would teach you, in nats. Use it to choose probes: the most informative offer is the one whose acceptance is genuinely uncertain, not the most ambitious nor the safest.
---

# t04_eig — Expected information gain

## 1. When to use it and when not
**Use it** every time you score candidate offers: it is the λ·EIG term of the personality's
rule. It is the RQ2 lever that turns an offer into an experiment.

**Do not use it** as the sole criterion: an offer can be very informative and bad at the same
time. The `econ` rule adds it to the expected utility, it does not replace it.

## 2. Inputs
| Input | Source |
|---|---|
| utility the offer gives the counterpart | your estimate of its utility |
| current round | gym |
| θ samples (rv, beta, T) | fox pool |

## 3. How to call it
```python
from tools.decision import expected_information_gain

info = expected_information_gain(u_to_counterpart, round_, theta, weights)
print(info["eig"], info["p_accept"])
```

## 4. How to read the output
`eig` in nats (expected reduction of entropy over `rv`), plus `p_accept` and the conditional
entropies. It is 0 when the answer is certain in either direction.

## 5. Failure modes
- **Degenerate belief**: if the posterior is already concentrated, the EIG is ~0 for everything.
  That is not a failure: there is nothing left to learn by this route.
- **Mis-estimated counterpart utility**: the EIG is computed over `u_to_counterpart`; if that
  number is wrong, the probe points at the wrong place.

## 6. Cost
Milliseconds: one pass over the θ samples.

## 7. Evidence
- `leahu2019auto` — frames elicitation as an action with an expected information gain that
  must exceed its cost. L2 of the report notes that **nobody** has applied Bayesian experimental
  design inside a negotiation session with a deadline: this tool is an implementation of that
  idea, not the replication of a result.

## 8. Minimal test
`pytest tests/test_decision_tools.py -k eig` — it is 0 when the answer is certain, peaks where
acceptance is uncertain and is never negative.
