---
name: f05-prior-from-the-corpus
description: >-
  CANDIDATE, not implemented: it would elicit with an LLM a prior over θ from the party's confidential corpus. Do not invoke it: the registry blocks it. It exists to document what is missing and how it would be validated.
---

# f05_case_reader_prior — Prior from the corpus (candidate)

## 1. When to use it and when not

**Do not use it.** Status `candidate`: it is not implemented and the registry raises an error if
you ask for it. Credentials now exist (DECISIONS D-025); what blocks it is that its validation
benchmark is not wired (D-007): without a validation report it cannot leave `candidate`.

Meanwhile, the preparation prior comes from f06 (flat) and the case information enters through
the party's own declared utility and the planner's strategy.

## 2. What inputs it needs and where they come from

| Input | Source |
|---|---|
| the party's confidential corpus | `cases/<case_id>/roles/<role_id>/` |
| public corpus | `cases/<case_id>/public/` |
| LLM credentials | `.env` through `agents/llm.py` |

## 3. How to call it

```python
# Blocked on purpose:
# FoxRegistry().create("f05_case_reader_prior")
# -> ValueError: f05_case_reader_prior is declared `candidate`
```

Validation: `python -m foxes.validate --fox f05_case_reader_prior`

## 4. How to read the output

When it exists: a `Posterior` over θ in the same common format as the rest.

## 5. Known failure modes

- **Leak risk**: it is the only fox that would read the confidential corpus. Its implementation
  will have to pass the canary test before being enabled.
- **Invention risk**: an LLM produces plausible numbers without evidence. That is why its
  validation plan contrasts it against a real human prior, not against itself.

## 6. Typical cost

Pending. It is the only fox with a token cost.

## 7. Evidence

- `konishi2026pref` and `lin2026dist` — lineage of the LLM-augmented Bayesian models.
- Validation plan: contrast the elicited prior against the prior assessed by a human expert in
  the Elmtree case (Raiffa, Figure 1: quartiles 250k / 475k over the buyer's RV). It is the only
  human benchmark available in the repository.

## 8. Minimal test

Not applicable until an implementation exists. The current test is that the registry rejects
it: `pytest foxes/tests/test_foxes.py -k registry_blocks`.
