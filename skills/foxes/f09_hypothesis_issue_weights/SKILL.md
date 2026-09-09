---
name: f09-weights-by-hypothesis
description: >-
  Estimates the counterpart's issue weights by asking how likely each observed offer is under each preference hypothesis. Use it from the second offer in multi-issue cases. It is the weight estimator with the best evidence in the catalog.
---

# f09_hypothesis_issue_weights — Issue weights over a hypothesis space

## 1. When to use it and when not

**Use it** with ≥ 2 issues and ≥ 2 observed offers. It is the weight estimator that beats the
control on both scoring rules, so in practice it is the first one to run.

**Run it alongside f02**, which uses a different mechanism (movement per issue vs. likelihood of
the offer). Disagreement between the two is signal, not noise: register a hypothesis and design a
probe for it.

**Do not use it** if the counterpart proposes at random or its utility is not additive: the Luce
likelihood assumes it proposes offers that are good *for itself*.

## 2. What inputs it needs and where they come from

| Input | Source |
|---|---|
| the counterpart's offers | gym trace |
| the domain's outcome space | `case.yaml` |
| calibrated dispersion sigma | `assets/calibration.json` |

## 3. How to call it

```python
from foxes.f09_hypothesis_issue_weights import HypothesisIssueWeights

fox = HypothesisIssueWeights()       # calibrated sigma is loaded from assets/
state = fox.init_state(domain)
state.data["counterpart_party"] = "gibsons"
post = fox.posterior(state)
```

Validation: `python -m foxes.validate --fox f09_hypothesis_issue_weights` · Calibration: `python -m foxes.calibrate --fox f09_hypothesis_issue_weights`

## 4. How to read the output

One parameter per issue, `w.<issue_id>`. The hypotheses also include the preference *direction*
per issue; if you need that part (for f07), extract it from the hypotheses with the most weight.

## 5. Known failure modes

- **Resolution of the hypothesis space**: 600 hypotheses over the simplex are enough for 3–5
  issues; with more issues the grid becomes sparse and the posterior artificially wide.
- **Badly calibrated sigma**: a low sigma assumes an almost optimal counterpart and produces
  overconfidence (0.41 coverage at 90 % with sigma = 0.03). The calibrated value is in `assets/`.
- **Cost**: it evaluates every hypothesis against the outcome space. It is the most expensive
  fox in the catalog; with large spaces raise `space_cap` carefully.

## 6. Typical cost

Hundreds of milliseconds per call (600 hypotheses × a space of up to 400 outcomes).

## 7. Evidence

- `konishi2026pref` §3.1 — hypothesis space over weights and evaluation functions with Bayesian
  updating; the linguistic likelihood of that work (which requires an LLM) is outside this version.
- `hindriks2008oppo` — the dispersion parameter that controls the aggressiveness of the update.

## 8. Minimal test

`pytest foxes/tests/test_foxes.py -k weights_rank` — same criterion as f02: the issue on which
the counterpart does not yield must receive more weight than the one on which it yields everything.
