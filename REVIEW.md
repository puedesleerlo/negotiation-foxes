# REVIEW.md — architecture review against the original intent

Date: 2026-09-06. Scope: the whole repository, read against the specification's idea — *a
multi-agent negotiation system whose every estimate about the counterpart comes from small,
calibrated models taken from papers (foxes), with two LLM party agents, strict corpus
isolation, sealed truth, falsifiable hypotheses and proper-scoring feedback.*

Every finding was confirmed by reading the code and, where it made a measurable difference, by
re-running the reference program (`parker_gibson-20260906-1038214a`) before and after the fix.
Findings are numbered F1–F12; the code, the catalog and DECISIONS.md refer to them by number.

| # | Finding | Severity | Status |
|---|---|---|---|
| F1 | The online foxes were hard-coded in the negotiator; the planner's plan was decorative | high | fixed |
| F2 | Hypotheses were registered at preparation and never updated during the rounds | high | fixed |
| F3 | No replay existed, although §10.1 makes deterministic replay the reproducibility test | high | fixed |
| F4 | The counterpart-utility estimate was an implicit `1 − u_own` with no declared scope | medium | fixed for single-issue; multi-issue wiring open |
| F5 | Foxes never received a response (`accept`/`reject`): only proposals reached them | medium | fixed |
| F6 | The control f06 drew fresh samples on every call: the baseline depended on call order | medium | fixed (f06 v0.1.1) |
| F7 | The control and experimental foxes entered the pool and the leave-one-out attribution | high | fixed |
| F8 | A reused preparation was not self-contained, and fox calls never left the negotiator's memory | medium | fixed |
| F9 | The decision memo and the belief rows did not record what they would need to be audited | medium | fixed |
| F10 | Token budgets are declared in `econ.yaml` and never enforced | low | open |
| F11 | f03 is validated *in-family* only: the synthetic counterparts follow the curve it fits | medium | open, documented |
| F12 | The foxes are the *lineage* of the papers, not reproductions of published results | info | documented |

## What the review confirmed is right

- **Every estimate about the counterpart comes from a fox.** No agent, tool or prompt produces
  a number about the other party outside `foxes/`. The LLM writes strategy and prose; the
  `ScriptedNegotiator` decides (DECISIONS D-030, D-037).
- **Isolation holds.** `PartyView` is built from the public corpus plus the party's own corpus
  only; the canary of each role is checked on every trace write; the sealed truth is read only
  by `feedback/`. The tests include an indirect leak through a file path.
- **The foxes enforce their own scope**, the registry refuses `candidate` foxes and marks
  `implemented` ones as experimental, and calibration and validation use disjoint splits.
- **Scoring is proper**: CRPS (energy form), log score via KDE, 50/90 coverage, Brier on
  hypotheses, all against the sealed truth.

## Findings in detail

### F1 — Online foxes hard-coded (fixed)
`ScriptedNegotiator` instantiated a fixed list of foxes regardless of the plan the planner had
produced and validated. The plan's `phase: online` tasks — the very thing the planner is for —
changed nothing at the table.
**Fix:** `gym.run.online_foxes_from_plan()` reads the adopted strategy; the negotiator receives
`online_foxes` and instantiates exactly those (plus the control). Foxes that are `implemented`
are created with `allow_experimental=True` and tracked in `self.experimental`; they never enter
the pool. `DEFAULT_ONLINE_FOXES` applies only when there is no plan. `adopt_strategy()` records
`online_foxes` and `foxes_source` in the trace.

### F2 — Hypotheses never updated during the rounds (fixed)
The hypothesis platform (§8) existed, the planner registered hypotheses, and the feedback
resolved them at the end — but nothing touched them between round 1 and the last round, so the
"living" part of the platform was dead.
**Fix:** `observe()` calls `update_hypotheses()` after every counterpart action. A hypothesis
about the counterpart's reservation value is judged against the bound its offer implies
(`_counterpart_rv_bound`); the threshold comes from the structured `threshold` field the planner
is now asked for, or from `hypotheses.thresholds.threshold_from_prose()` as a conservative
fallback (English and Spanish phrasings, negations handled), or from `interval`. Updates are
saved with the hypotheses (`hypotheses_saved` event, `n_updates_during_rounds`). Measured on the
reference program: parkers 1 update during the rounds; gibsons 0 — their hypotheses are of the
form "rv ≤ X" about a seller, which a seller's *offers* cannot decide. That is correct behaviour,
and the trace says so.

### F3 — No replay (fixed)
§10.1 defines replay as the reproducibility test and §10.2 makes `events.jsonl` the source of
truth with regenerable DuckDB tables. Neither existed.
**Fix:** `gym/replay.py` re-derives every decision of a recorded program from the same inputs
(case, seed, personalities, adopted strategies, recorded counterpart actions) and reports
mismatches; `--rebuild-db` rebuilds DuckDB from the JSONL. Result on the reference program:
26/26 decisions reproduced; DuckDB `{events: 175, actions: 26, fox_calls: 112}`.

### F4 — Counterpart utility implicit (fixed for single-issue; multi-issue open)
Every online fox needs "what this offer is worth *to the counterpart*". The negotiator computed
it as `1 − u_own` silently. That is exact for a single-issue distributive case with linear
utilities, and wrong for anything else — and a wrong estimate here corrupts f01 and f03 without
any symptom in the trace.
**Fix:** `counterpart_utility_model(view, own_utility)` makes the assumption explicit, is recorded
in every decision memo, and raises `NotImplementedError` for multi-issue cases. **Open:** the
multi-issue estimate must be built from the f02/f09 weight posteriors; that wiring does not exist
yet and E1 currently supports single-issue cases only.

### F5 — Foxes never saw responses (fixed)
The gym only forwarded `propose` actions to the observer. f04 (acceptance boundary) declares
that it needs responses; it could never receive one in a program, so its "out of scope" verdict
in real programs was an artefact of the wiring, not a property of the data.
**Fix:** `observe()` emits the implied `reject` of the party's own previous offer (and the
counterpart's `accept` when it happens) before the counterpart's proposal. `foxes/base.py`
`counterpart_offers()` now filters `action == "propose"` — the first version of this fix let a
party's own offer be counted as the counterpart's and stalled the program to the deadline; a
test (`test_counterpart_offers_excludes_responses`) pins it.

### F6 — Non-idempotent control (fixed)
f06 drew new samples on each call, so two calls on the same state gave different posteriors.
Since every comparison in the system is *against the control*, its numbers depended on how many
times it had been called.
**Fix:** f06 v0.1.1 keys its RNG on (seed, round, number of observations); a test checks
idempotency on the same state.

### F7 — Control and experimental foxes in the pool (fixed)
The equal-weight pool and the leave-one-out attribution included f06 and experimental foxes.
That understates every validated fox's contribution (the flat prior dilutes the pool) and makes
"does the pool beat the control?" a comparison of the control against itself.
**Fix:** pool membership = `scope_ok ∧ not control ∧ not experimental`, recorded per fox as
`in_pool` in the memo; `feedback.metrics` computes `control_crps`/`control_log_score` from the
control's own quantiles. Corrected attribution on the reference program: f03 +1.356 (gibsons)
/ +0.628 (parkers); f01 −0.470 / −0.249 (Δ log score of the pooled belief, leave-one-out). The
sign of f01's contribution — negative — is the measurable face of the bias documented in D-035.

### F8 — Reuse not self-contained; fox calls lost (fixed)
`--reuse-preparation` pointed a program at another program's directory; replaying the new
program then needed the old one, and the strategy on disk did not match the trace. Fox calls
were kept in each negotiator's in-memory registry and never emitted, so the trace could not show
where a belief came from (rule 12 of §6.5).
**Fix:** reuse copies `strategy.json`, `plan.json`, `strategy.md` and `hypotheses.jsonl` into the
new program and records `preparation_reused.source`; `drain_fox_calls()` emits `fox_call` events
after every round; replay follows the recorded source.

### F9 — Audit trail in the memo and the belief rows (fixed)
The decision memo did not say which foxes were in the pool or which counterpart-utility model
was used; the belief rows stored the pooled quantiles only, so the feedback could not compute
the control's score or any per-fox diagnostic from the trace.
**Fix:** the memo records `counterpart_utility_model` and per-fox `in_pool`; the belief rows
store `per_fox_quantiles` (in-scope foxes) and `control_quantiles`.

### F10 — Budgets declared, not enforced (open)
`econ.yaml` declares token budgets for preparation, per round and per researcher task. Only the
researchers' wall-clock budget is enforced. Costs are recorded (§10.2) but nothing stops a step
that exceeds its budget. Left open: enforcing it requires deciding what a party does when it runs
out mid-round, which is a design decision for the owner.

### F11 — In-family validation of f03 (open, documented)
The synthetic counterparts follow the Faratin–Sierra–Jennings curve that f03 fits. Its validation
shows that the estimator recovers the parameters of its own model, not that real counterparts
follow it. The scope check (goodness of fit) is what protects a program from a counterpart that
does not. Documented in the validation reports, `foxes/synthetic.py`, `foxes/validate.py` and the
catalog; the out-of-family evidence has to come from real programs.

### F12 — Lineage, not reproduction (documented)
Each fox implements the mechanism the cited paper describes, as reconstructed from the survey and
the extracted text, with our own discretisation and calibration. No fox reproduces a published
number, and none should be cited as if it did. Stated in the catalog header and in the fox
docstrings.

## Also changed by the review

- **Language.** The project's working language is English everywhere — code, comments, prompts,
  reports, catalog, skills, documentation (DECISIONS D-048, supersedes D-013). The prose
  threshold fallback still understands Spanish phrasings because recorded preparations exist in
  Spanish.
- **Tests.** 117 pass (`pytest foxes/tests tests -q`), including new ones for chronology of
  implied observations, attribution excluding the control, replay from the trace, DuckDB rebuild,
  plan-driven online foxes, thresholded and prose-fallback hypothesis updates, self-contained
  reuse, f06 idempotency and f04 receiving responses.

## What remains beyond the findings

- Multi-issue counterpart utility from f02/f09 (F4), which is what would make t06/t07 and the
  weight foxes reachable in a program.
- f05 (`case_reader_prior`) is still `candidate`: it needs its validation benchmark (Raiffa's
  human prior) wired before it can be promoted.
- The knowledge base's GraphRAG (D-016) is still replaced by the FTS5 index.
- The utility–impasse frontier needs a case with a narrow or uncertain ZOPA (D-032).
