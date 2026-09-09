---
name: f02-weights-from-concession
description: >-
  Estimates the weight the counterpart gives each issue by looking at which ones it concedes on first. Use it from the third observed offer in multi-issue cases. Returns a distribution over the simplex, not a point vector.
---

# f02_concession_issue_weights — Issue weights from the concession ratio

## 1. When to use it and when not

**Use it** with ≥ 2 issues and ≥ 3 consecutive offers from the counterpart.

**Do not use it** in a single-issue case (the weights are undefined), nor when the counterpart
has repeated the same offer: without movement there is no concession ratio to measure.

**Use it alongside f09, never instead of it.** They measure the same thing with independent
mechanisms (f02, the movement per issue; f09, the likelihood of each offer). If their medians
differ by more than a threshold, register a hypothesis and design a MESO probe that
discriminates between them (protocol §6.5, rule 4).

## 2. What inputs it needs and where they come from

| Input | Source |
|---|---|
| consecutive offers from the counterpart | gym trace |
| ranges and possible values of each issue | `case.yaml` |

It does not need to know the counterpart's value functions: it measures movement on the raw
scale of each issue.

## 3. How to call it

```python
from foxes.f02_concession_issue_weights import ConcessionIssueWeights

fox = ConcessionIssueWeights()      # the calibrated concentration is loaded from assets/
state = fox.init_state(domain)
state.data["counterpart_party"] = "gibsons"
for obs in observations:
    state = fox.update(state, obs)
post = fox.posterior(state)
print(post.summary()["w.price"])
```

Validation: `python -m foxes.validate --fox f02_concession_issue_weights` · Calibration: `python -m foxes.calibrate --fox f02_concession_issue_weights`

## 4. How to read the output

One parameter per issue, `w.<issue_id>`, summing to 1 in every sample. Use
`post.mean("w.price")` for the expected weight and the interval to know whether you can really
tell two issues apart: if the intervals of `w.price` and `w.closing` overlap, you cannot.

## 5. Known failure modes

- **Scales not comparable across issues**: the concession ratio normalises by the issue range;
  if one issue has three values and another is continuous, movement does not mean the same.
- **Strategic concession**: a counterpart that concedes first on what matters most to it, to
  simulate flexibility, inverts the estimated order. The assumption is the lineage's, not a law.
- **Uncalibrated overconfidence**: without the file `assets/calibration.json` the 90 % coverage
  drops to 0.63. If you delete that file, the fox is no longer calibrated.
- **It does not beat the control's log score** on the synthetic dataset, because there the true
  weights are drawn from Dirichlet(1) and the control *is* that distribution. It beats the CRPS.

## 6. Typical cost

Milliseconds. 400 bootstrap resamples. No LLM.

## 7. Evidence

- `baarslag2015lear` §5.3 — collects the estimator: concession happens first on the least
  important issues, and the normalised concession ratio gives `w_i ∝ 1 − c_i` (lineage Jonker et
  al., Carbonneau and Vahidov, Niemann and Lang).

## 8. Minimal test

`pytest foxes/tests/test_foxes.py -k "weights_rank or f02"` — with a counterpart that yields
everything on `employment` and nothing on `price`, `price` must receive more weight; and in a
single-issue domain the fox must declare itself out of scope.
