---
name: f03-time-dependent-tactic
description: >-
  Fits the counterpart's concession curve and estimates its concession shape (beta). Use it from the fourth observed offer when the counterpart seems to move with time. It also reports rv and deadline, but those two are NOT validated: do not use them as if they were.
---

# f03_time_concession_regression — Regression of the time-dependent tactic

## 1. When to use it and when not

**Use it** with ≥ 4 observed offers *and* when the counterpart's curve can be fitted: the fox
checks the RMSE of the best fit and declares itself out of scope above 0.10. That automatically
discards tit-for-tat or erratic counterparts.

**Use it for `beta` only.** That is the only validated output. The estimate of `T` (deadline)
does not beat the control: rv and T are structurally coupled and the deadline is not recovered
from the offer curve. If you need the deadline, treat it as unknown and marginalise it.

**It complements f01**: f01 bounds rv below the lowest offer; f03 explains the shape of the
trajectory. Against Boulware, f03 is the one that can say something.

**In-family caveat (REVIEW F11)**: the validation runs on synthetic counterparts that follow the
very curve f03 fits, so it shows that the estimator recovers the parameters of its own model. The
goodness-of-fit scope check is what protects a program from a counterpart that does not follow it.

## 2. What inputs it needs and where they come from

| Input | Source |
|---|---|
| utility of each counterpart offer to the counterpart | single-issue: the gym's complementary utility; multi-issue: f02/f09 give the weights, without them do not use f03 |
| round number of each offer | gym trace |

## 3. How to call it

```python
from foxes.f03_time_concession_regression import TimeConcessionRegression

fox = TimeConcessionRegression()
state = fox.init_state(domain)
state.data["counterpart_party"] = "gibsons"
post = fox.posterior(state)          # parameters: rv, beta, T
beta_median = post.quantile("beta", 0.5)
```

Validation: `python -m foxes.validate --fox f03_time_concession_regression` · Calibration: `python -m foxes.calibrate --fox f03_time_concession_regression`

## 4. How to read the output

Three parameters: `rv`, `beta`, `T`. **Only `beta` is validated.** `beta < 1` means Boulware
(concedes at the end), `beta ≥ 1` Conceder. The warnings include the rv–T correlation of the
posterior: if it is high (usual), those two parameters are not separated by the data.

## 5. Known failure modes

- **Identifiability ridge**: many (rv, T) combinations explain the same curve. Version 0.1.0
  used a residual bootstrap and gave 0.12 coverage at 50 %: it fitted well and lied about its
  uncertainty. 0.2.0 uses a grid posterior and gives 0.66/0.80.
- **A counterpart that changes tactic mid-session**: the constant-parameter assumption breaks
  and the fit averages two regimes.
- **Residual overconfidence**: 90 % coverage is 0.80, below nominal. With counterparts whose
  behaviour only loosely resembles the curve, the interval falls short.

## 6. Typical cost

Tens of milliseconds. Grid of 21×15×16 = 5,040 combinations. No LLM.

## 7. Evidence

- `faratin1998nego` — time-dependent decision functions (Boulware/Conceder).
- `baarslag2015lear` §3.6 — the functional form used: `u(t) = Pmin + (Pmax − Pmin)(1 − F(t))`
  with `F(t) = k + (1 − k)·t^(1/e)`. §5.2 documents the rv–deadline coupling.

## 8. Minimal test

`pytest foxes/tests/test_foxes.py -k f03_recovers` — on a curve generated with known
parameters, the 90 % interval must contain the true rv.
