---
name: f06-uninformed-prior
description: >-
  Mandatory control: a flat prior over the declared support of each parameter. Run it ALWAYS, in every program and for every parameter of θ. Without it nothing can be said about how much information the other foxes add. It is never a member of the pool.
---

# f06_uninformed_prior — Uninformed prior (control)

## 1. When to use it and when not

**Always.** It has no scope conditions. Rule 2 of protocol §6.5 makes it mandatory: no
component of θ is estimated with a single fox, and f06 runs in addition to the others to
measure how much the information adds.

**It never enters the pool** (REVIEW F7): it is the baseline the pool's CRPS and log score are
measured against. If for some parameter no fox is left in scope, f06 is what remains and **that
must be declared in the memo**: it means nothing is known about that parameter beyond its support.

## 2. What inputs it needs and where they come from

| Input | Source |
|---|---|
| declared support of each parameter | `case.yaml` and the fox's `DEFAULT_SUPPORT` |

## 3. How to call it

```python
from foxes.f06_uninformed_prior import UninformedPrior

control = UninformedPrior(params=["rv"])
post = control.posterior(control.init_state(domain))
```

Validation: `python -m foxes.validate --fox f06_uninformed_prior`

## 4. How to read the output

Uniform samples over the support (Dirichlet(1) when the parameters are weights). Since 0.1.1
the sampler is keyed on (seed, round, number of observations): two calls on the same state
return the same posterior (REVIEW F6).

## 5. Known failure modes

- **Mis-declared support**: if the range in `case.yaml` does not contain the true value, the
  control's coverage drops and *every* fox inherits the bias. That is why its validation report
  exists: it verifies that the support contains θ.
- **Reading it as a prediction**: it is not one. It is the reference the rest is measured against.

## 6. Typical cost

Microseconds.

## 7. Evidence

No paper: it is the experimental control the project's protocol demands (§6.5, rule 2).

## 8. Minimal test

`pytest foxes/tests/test_foxes.py -k f06` — its 90 % interval must cover roughly 90 % of true
values sampled from the support, and two calls on the same state must agree.
