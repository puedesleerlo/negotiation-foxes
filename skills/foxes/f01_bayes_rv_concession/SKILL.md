---
name: f01-reservation-value-from-concession
description: >-
  Estimates the counterpart's reservation value (rv) from its concession sequence, and returns
  a distribution, not a number. Use it during the negotiation, from the second observed offer,
  when the counterpart is moving towards you. Do not use it if the counterpart has conceded
  nothing: in that case it will tell you, and its output must not enter the pool.
---

# f01 — Reservation value from concession

## 1. When to use it and when not

**Use it** when the three scope conditions hold:
- you have observed **≥ 2 offers** from the counterpart;
- the counterpart **concedes with a trend**: the mean of its first offers is at least 0.03 (on
  its utility scale) better for it than the mean of the last ones;
- the counterpart's rv does not change within the session.

**Do not use it** — or discard its output — when:
- the counterpart has not moved (hardliner): the fox sets `scope_ok = False` and its posterior
  collapses to the hard bound rv ≤ (lowest observed offer), which is uninformative;
- the counterpart goes up and down without a trend (erratic or dependent on your own
  behaviour): the monotone-concession assumption does not hold;
- you are in preparation and there are no offers yet: use f05 (if available) or f06.

## 2. What inputs it needs and where they come from

| Input | Source |
|---|---|
| the counterpart's offers per round | gym trace (`Observation.offer`) |
| utility of each offer **to the counterpart** | estimated by your party: in a single-issue case the gym uses the complementary linear utility (`counterpart_utility_model`); in multi-issue cases f02/f09 give the weights, and without them do not use f01 |
| round number and protocol horizon | `case.yaml` (`max_rounds`) |

f01 does **not** receive the counterpart's corpus or the sealed truth. Its input is the table only.

## 3. How to call it

```python
from foxes.f01_bayes_rv_concession import BayesRvConcession
from foxes.base import Observation

fox = BayesRvConcession()
state = fox.init_state(domain)
state.data["counterpart_party"] = "gibsons"
state = fox.update(state, Observation(round=3, party="gibsons", offer=offer, action="propose",
                                      est_utility_to_proposer=0.82, max_rounds=20))
post = fox.posterior(state)
```

CLI: `python -m foxes.validate --fox f01_bayes_rv_concession` reproduces its validation.

## 4. How to read the output

A `Posterior` with the parameter `rv` on the counterpart's utility scale (0 = its worst
outcome, 1 = its best). Use `post.quantile("rv", 0.25)` to decide with optimism (the
personality's τ: a *low* rv is the favourable scenario) and `post.entropy("rv")` to track how
much the belief has narrowed. Always check `post.scope_ok` and `post.warnings` before using it:
they carry the correlation with the deadline and the scope notices.

## 5. Known failure modes

- **Hardliner disguised as concession**: if the counterpart alternates between almost
  equivalent offers, the utility range is non-zero without any real concession. That is why
  scope is measured with a trend (first third vs last third), not with the range.
- **Boulware within a short horizon**: it concedes almost everything at the end; within 24
  rounds the fox cannot separate its rv from the control (see the validation report). Its output
  is still valid but contributes little: it is the case where f03 should carry the weight.
- **Unknown deadline**: rv and deadline are coupled. The fox marginalises over a grid of
  possible deadlines instead of assuming the protocol horizon is the counterpart's; that widens
  the posterior on purpose.
- **A counterpart that never concedes below its target** leaves the hard bound high, and the
  posterior overestimates rv. Measured on the Parker-Gibson reference program: the pooled belief
  sits above the truth and f01's leave-one-out contribution is negative (DECISIONS D-035, D-047).
- **Error in the estimated counterpart utility**: it propagates in full. The validation is done
  in the *oracle* condition (true utility), so the real error will be larger.

## 6. Typical cost

Milliseconds. Grid of 201 points × 16 deadlines. No LLM.

## 7. Evidence

- `zeng1998baye` — Bayesian learning over a hypothesis space of the RV (Bazaar). It is the
  origin of the mechanism: discrete hypotheses about the RV updated with the observed offers.
- `baarslag2015lear` §4.1 and §5.1.1 — operational formulation: the counterpart stops conceding
  near its reservation value and usually does so as its deadline approaches. §5.2 documents the
  rv–deadline coupling that forces the marginalisation.
- Lineage, not reproduction (REVIEW F12): the discretisation and the likelihood are ours.

## 8. Minimal test

`pytest foxes/tests/test_foxes.py -k f01` — checks that (a) with a counterpart conceding
towards a known rv the posterior median falls inside the declared interval, (b) with a
hardliner the fox declares `scope_ok = False`, and (c) the posterior never puts mass above the
lowest observed offer.
