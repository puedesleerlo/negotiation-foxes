# DECISIONS.md

Decision log. Format: date, decision, discarded alternatives, how to revert it.
Decisions marked ⚠ await confirmation from the project owner but do not block the work.

---

### D-001 · 2026-09-05 · Repository root in `comparator/`, not in `negotiation-foxes/`
The specification (§14) names the root `negotiation-foxes/`. The repository already exists as
`~/repos-new/comparator` with the owner's material inside. The §14 structure is built *inside*
that directory instead of creating a nested folder.
**Discarded:** creating `comparator/negotiation-foxes/` (nests without gain); renaming the
user's directory (affects paths outside my view).
**Revert:** `mv` the structure into a subfolder; nothing depends on the absolute path.

**Superseded on 2026-09-18** by D-049: the working directory is now `~/repos-new/negotiation-foxes`.

### D-002 · 2026-09-05 · `kosmos/` → `knowledge/kosmos/`
To match the §14 structure without duplicating. The original PDF was not modified.
**Revert:** `mv knowledge/kosmos kosmos`.

### D-003 · 2026-09-05 · `cases/case_1/` → `cases/elmtree_house/source/`
The case is Elmtree House (Raiffa, ch. 3). `case_1` does not say what it is; `source/` separates
the source material from the derived artifacts (`case.yaml`, `public/`, `roles/`).
**Revert:** `mv cases/elmtree_house/source/*.pdf cases/case_1/`.

### D-004 · 2026-09-05 · Python 3.11 with `uv`, dependencies without packaging the project
`uv venv --python 3.11` + `uv pip install`. The project is used as a set of modules with
`PYTHONPATH` at the root, not as an installable package: there are no external consumers and it
avoids build friction on every change.
**Discarded:** `uv pip install -e .` (fails without a package layout and adds nothing here); the
system's Python 3.14 (outside the support range of several scientific dependencies).
**Revert:** add `[build-system]` and a `src/` layout.

### D-005 · 2026-09-05 · Own outcome-space mathematics; NegMAS optional
The specification (§13) suggests NegMAS as the default for domains, additive utilities, Pareto
frontier, Nash, Kalai–Smorodinsky and for generating the synthetic dataset. Implemented in-house
in `foxes/domain.py` (~300 lines): additive utilities, exact frontier by enumeration/sampling,
Nash, KS, counterpart families and the alternating-offers protocol.
**Reason:** it is exact, testable mathematics, we need it deterministic and seeded, and it
avoids a large dependency whose agent model we will not use (the LLM agent loop is our own
anyway, as §16 already foresaw).
**Discarded:** NegMAS as a hard dependency (installation risk and coupling of the protocol).
**Revert:** `uv pip install negmas` and use `foxes/domain_negmas.py` as an alternative
implementation behind the same interface; to be cross-checked if it installs cleanly.

### D-006 · 2026-09-05 · ⚠ Models per agent role (superseded by D-025)
The initial design named `claude-opus-5` for planner and negotiator and `claude-haiku-4-5` for
mechanical researchers, through the official Anthropic SDK with the tool runner for planner and
researchers and a manual loop for the negotiator (the gym must intercept every turn).
**Discarded:** Claude Agent SDK (brings a filesystem harness that would break corpus
isolation); Managed Agents (the state and the trace are ours).
**Status:** superseded — the owner chose Kimi K3 through an OpenAI-compatible endpoint (D-025).

### D-007 · 2026-09-05 · f05 (`case_reader_prior`) stays `candidate` until it can be validated
Originally: no credentials in the environment. "Zero mock-ups" rule: an unimplemented fox is
declared `candidate` and not used. Part 1 is accepted without it (f01–f04 + f06 + f07).
**Update:** credentials exist since D-036; what still blocks f05 is that its validation benchmark
(Raiffa's human prior, Figure 1) is not wired. Without a validation report it cannot be promoted.

### D-008 · 2026-09-05 · No Unpaywall with the owner's personal e-mail
Unpaywall requires an e-mail in the URL of every query. The owner's e-mail is not sent to an
external service without explicit permission. OpenAlex (`best_oa_location`) and arXiv are used,
which give equivalent coverage without a personal identifier.
**Revert:** the owner authorises an e-mail (own or a project contact) and `--unpaywall-email`
is enabled in `kb/fetch_papers.py`.

### D-009 · 2026-09-05 · Double extraction of Kosmos references
The report's list is truncated. `references.jsonl` distinguishes `ref_kind: listed` (formal
entry with DOI/URL) from `ref_kind: inline` (work named in the body, resolved by search). Both
carry the `kosmos_claim` they support.
**Discarded:** ingesting only the formal entries (would lose ~40 works that ground L1 and L2).
**Revert:** re-run the ingestor on the full export if it appears (⚠ Q3).

### D-010 · 2026-09-05 · L3 is ingested but produces no foxes
L3 deals with credit assignment in ensembles; §17 declares it out of E1's scope. Its papers
enter the base with `relevance: out_of_scope_e1` so that later credit-based weighting has the
evidence ready.
**Revert:** change `relevance` and add entries to the catalog.

### D-011 · 2026-09-05 · Equal-weight linear pooling as the default
Rule 3 of protocol §6.5, further supported by the L3 evidence (forecast hubs: the equal-weight
ensemble is hard to beat; the median beats the mean).
**Note:** the individual posteriors are stored so other combinations can be recomputed. The
*median pool* is also implemented as a registered variant, not activated.

### D-012 · 2026-09-05 · ⚠ The Elmtree case splits into two artifacts (partly superseded by D-024)
`elmtree_house_raiffa`: validation case as it stands in the book — one side, one issue, partial
truth (seller RV = $220,000 explicit; buyer RV ≥ $325,000 from the realised outcome; Raiffa's
human prior over the buyer's RV as f05's external benchmark).
`elmtree_house_e1`: a two-sided, multi-issue case to be built from the same chapter, with
per-parameter provenance (textual vs constructed), to run E1 end to end.
**Status:** the construction half was superseded when the owner provided Parker-Gibson (D-024);
the validation half stands.

### D-013 · 2026-09-05 · Documentation in Spanish, code in English (superseded by D-048)
The owner's original rule (§0), applied also to the `SKILL.md` files.
**Status:** superseded — the owner asked for English as the single language (D-048).

### D-014 · 2026-09-05 · No `git init` for now
The directory is not a git repository and the owner did not ask for one. `DECISIONS.md` carries
the decision trail meanwhile. Recommended before Part 2 (programs and their seeds gain a lot
from versioned history).
**Revert:** `git init && git add -A`.

### D-015 · 2026-09-06 · `kb/` and `scripts/` are added to the §14 structure
The specification's structure has nowhere for the Part 1 code to live (ingestion, resolution,
download, database, annotation). Two directories are added: `kb/` (knowledge-base package) and
`scripts/` (one command per experiment).
**Revert:** move `kb/` under `knowledge/` to group code and data.

### D-016 · 2026-09-06 · The KB's GraphRAG is postponed; FTS5 index instead
Task 1.4 asks for the knowledge base's GraphRAG. LightRAG needs an LLM and an embedding model,
and there were no credentials at the time. Instead, `papers.sqlite` includes a full-text index
(FTS5 with stemming) over the extracted sections, which covers the query Part 1 really needs:
finding the passage that grounds a fox.
**Discarded:** local embeddings (a heavy dependency for a benefit that only pays off when the
agents query the KB).
**Revert:** build the GraphRAG over `knowledge/extracted/` without touching anything else; the
FTS5 index can coexist.

### D-017 · 2026-09-06 · Explicit promotion rule to `validated`
A fox becomes `validated` if (a) the coverage of its 90 % interval is between 0.80 and 0.97 on
the validation split, and (b) it beats the control f06 on at least one proper scoring rule
(CRPS or log score), with the deficit on the other declared in its `validation_summary`.
**Reason:** without a written rule, "validated" becomes an opinion. With it, f04 stays
`implemented` and f03 is validated for `beta` only, which is what the evidence supports.
**Revert:** change the threshold in `catalog/rubric.md` and re-grade; the reports already have
the numbers to do it without re-running anything.

### D-018 · 2026-09-06 · Calibration and validation on disjoint partitions
40 % of the episodes (by hash of `episode_id`) are reserved for calibrating the dispersion
parameters; the remaining 60 % for validation. No number in a validation report comes from data
seen while calibrating.
**Revert:** `foxes.validate.split_of`, a single place.

### D-019 · 2026-09-06 · f03 moves to a grid posterior (v0.2.0)
Version 0.1.0 estimated (rv, e, T) by least squares with a residual bootstrap. It fitted the
curve well and lied about its uncertainty: measured coverage 0.12 at 50 % and 0.36 at 90 %. The
residual bootstrap does not capture the ridge of (rv, T) combinations that explain the same
curve almost equally well, which is precisely the identifiability problem the survey documents
(§5.2). 0.2.0 uses a grid posterior with a Gaussian likelihood: coverage 0.66/0.80 and CRPS of
`beta` 0.34 against 1.07 for the control.
**Revert:** the bootstrap mode is still in the code (`n_bootstrap`), unused.

### D-020 · 2026-09-06 · f04 stays `implemented`, and why is documented
Two measured findings: (1) no out-of-sample prediction meets its scope condition — it needs ≥ 4
responses *with variation*, and in this protocol the counterpart rejects the first four offers
almost always; (2) operating out of scope, its Brier (0.049) does not beat predicting the base
rate (0.043). The calibrated population prior improved a lot (Brier 0.075 → 0.049) but not enough.
**Plan v0.2:** derive `p_accept` from the pool's posterior over (rv, beta) as
`P(u_cp(offer) ≥ target_cp(t))`, instead of estimating its own boundary with one session's data.
It is the same quantity, estimated where there is information. (Done as a tool: D-026.)

### D-021 · 2026-09-06 · Changes to the specification's seed catalog (§6.3)
`f02_freq_issue_weights` → `f02_concession_issue_weights` (the evidence describes a concession
ratio, not a frequency count). `f04_accept_boundary_np` → `f04_accept_boundary_kde` (the
technique is fixed). Added `f09_hypothesis_issue_weights` (second weight estimator with an
independent mechanism, which rule 2 of the protocol requires) and `f10_impasse_risk`
(`candidate`, so as not to lose the corpus's only quantitative evidence about the cost of being
ambitious). Details and reasons in `catalog/rubric.md`.

### D-022 · 2026-09-06 · No PDF is downloaded for references without confirmed metadata
A title search without confirmed author and year returns similar but different works: in an
early run a PDF on another topic was attached to "Lee et al. 2024". Attaching the wrong PDF to a
reference is worse than having no PDF, because it contaminates the catalog's evidence.
Exception: the direct links carried by the Kosmos entry itself, which are authoritative.

### D-023 · 2026-09-06 · Validation runs in the *oracle* condition
The foxes receive the true utility each offer gives the counterpart. This isolates the
estimator's error from the error of estimating the counterpart's utility, which is a separate
problem (and what f02 and f09 estimate). Declared in every report: **in a real program the two
errors add up**, so these numbers are an optimistic bound.
**Pending:** repeat the validation of f01 and f03 with weights estimated by f02/f09 to measure
how much they degrade in chain.

### D-024 · 2026-09-06 · E1's case is Parker-Gibson (PON), and it is single-issue
The owner provided the real two-sided case: `cases/parker_gibson/`, with the confidential
instructions of both parties. It replaces the plan to build a second side for Elmtree (D-012's
construction half is superseded; `elmtree_house` stays as a validation case with Raiffa's human
prior).

**Sealed truth, policy `explicit`:** both reservation values are textual. Each party's
confidential instructions state its limit in so many words: the seller's floor is an offer it
already has in hand, the buyer's cap is a stipulated maximum. The values themselves live only in
the local case bundle and are not reproduced here (see the case's NOTICE.md).

**A scope consequence to read as confirmation, not as a problem:** the case is distributive
and single-issue and explicitly forbids introducing other issues or options. Hence f02 and f09
(issue weights) and the tools t06/t07
(MESO, logrolling) **declare themselves out of scope on their own**, with the mechanism already
tested. The applicable foxes are f01, f03, f06 and f07.

**The asymmetry that makes the case interesting:** the seller's BATNA is a concrete number
(an offer from another buyer) and the buyer's is diffuse (a costlier and less attractive
construction alternative). Estimating the buyer's rv is structurally harder than the seller's,
and that should show in the per-party calibration.

**Rights:** the material belongs to Harvard's Program on Negotiation and cannot be
redistributed. `cases/parker_gibson/NOTICE.md` fixes the rule: extracted text for local use only,
no report or dashboard includes the material, and the case stays out of any distributed package.

### D-025 · 2026-09-06 · Model provider: Kimi K3 through an OpenAI-compatible endpoint
The owner chose Kimi K3 with credentials in `.env` (`LLM_API_KEY`, `LLM_ENDPOINT`, and optional
`LLM_MODEL`). The whole system talks to `agents/llm.py` and only to it: changing provider means
changing that file, not the agents.
**Discarded:** the Anthropic SDK and the earlier design of D-006, which is annulled.
**Status:** `.env` was empty at first; `python scripts/check_llm.py` lists what is missing, tests
the endpoint and makes a smoke call. Without credentials no agent runs: the client raises
`CredentialsMissing` instead of simulating replies. Credentials were verified later (D-036).

### D-026 · 2026-09-06 · Acceptance probability is derived from the belief, not estimated separately
Implements f04's v0.2 plan (D-020) but as a **tool**, not a fox:
`tools.decision.p_accept_from_belief` evaluates P(u_cp(offer) ≥ target_cp(t)) over the
(rv, beta, T) samples that f01 and f03 already produced. It needs no acceptance data of its own —
which is precisely what a session does not have — and introduces no new model to calibrate.
**Revert:** f04 stays in the catalog as `implemented` in case its direct approach proves useful
with another protocol that does reveal informative rejections.

### D-027 · 2026-09-06 · Direction of optimism τ, written down so it is not inverted
`rv` is the *counterpart's* reservation utility: a low rv means it accepts offers that are worse
for it, i.e. better for us. That is why "optimism τ = 0.75" is implemented as the **quantile
1−τ = 0.25** of rv. Documented in `econ.yaml` and in `Personality.optimistic_theta` because it is
the easiest sign error to make in the whole system.
Measured effect on the case: with τ = 0.5 and λ = 0 the seller asks 16,500 (almost its own
reserve); with τ = 0.75 and λ = 0.25 it asks 20,500–22,500. It is the pathology RQ2 attacks,
reproduced on the bench.

### D-028 · 2026-09-06 · `tools/` as the code package for the tools
`skills/tools/<tool_id>/` holds the usage instructions; the code lives in `tools/decision.py`
and `tools/integrative.py`. Extends D-015.

### D-029 · 2026-09-06 · The contingent clause is priced at the midpoint of the beliefs
Pricing the bet at one's own belief gives zero expected value for oneself: the surplus comes from
the *difference* between beliefs. With the midpoint, both parties see `stake × difference ÷ 2`.
**Warning recorded in the skill:** L2 of the report says explicitly that contingent contracts are
mentioned but have no computational treatment or empirical quantification in the retrieved
literature. The tool implements a criterion; it does not replicate a result.

### D-030 · 2026-09-06 · A deterministic negotiator before the LLM agent
`ScriptedNegotiator` does everything the model agent will do except write: it observes, updates
the foxes, pools, registers hypotheses on disagreement, scores candidates with the personality's
rule and writes the `decision_memo`. It allows complete programs without credentials and, above
all, keeps the decision core **separate from the text**: when the LLM comes in it replaces the
writing of the memo and the message at the table, not the choice of the action. It is the
operational form of rules 7 and 8 of the protocol (§6.5): the model cannot override the foxes
because it is not the one deciding.

### D-031 · 2026-09-06 · Three semantic corrections the first complete program uncovered
Running for real revealed three errors no unit test would have found:

1. **Walking away in round 1.** The walk-away rule looked at the snapshot of the current round,
   where the counterpart still asks for almost everything whatever its reservation value. It is
   now judged with the candidates evaluated at the end of the horizon.
2. **Accepting almost at one's own reserve.** The value of waiting was also computed with the
   current round, so "continuing" always looked bad. It is now computed over the future rounds.
3. **τ and λ without effect.** The aspiration was a constant (0.85) and the concession policy,
   which interpolates between aspiration and reserve, completely masked the levers. The
   aspiration now **comes from the belief**: what the agent believes reachable at the end of the
   horizon under its optimistic quantile. With that, τ moves the whole policy, which is what is
   to be studied.

In addition, the value of waiting uses the **optimistic** acceptance probability: the
personality's optimism is not only a way of choosing an offer, it is a way of valuing waiting.
With τ = 0.5 both coincide and the rule is neutral again.

### D-032 · 2026-09-06 · The lever sweep is asymmetric
If τ rises on both sides at once, what is measured is the strategic interaction, not the lever.
`gym.batch --focal <role>` sweeps τ and λ on one party and keeps the other at the baseline
(τ = 0.5, λ = 0). Measured on Parker-Gibson with a 20-round horizon, sweeping the seller:
τ = 0.50 → 23,250 USD; 0.65 → 27,375; 0.75 → 35,375; 0.90 → 33,000–35,500 with 15.75 rounds on
average. The impasse rate is 0 across the sweep.

**What that zero means:** in this case the ZOPA covers 45 % of the price range and the horizon
leaves plenty of slack, so the impasse arm of the frontier **does not appear**. Tracing the full
utility–impasse frontier needs a case with a narrow or uncertain ZOPA, not more sweeping of this
one. Recorded as a limitation, not as a result.

### D-033 · 2026-09-06 · Python's `hash()` is out of the seeds
The per-role seed offset used `hash(role_id)`, which is randomised per process: two runs of the
same seed in different processes diverged. A stable hash (SHA-256) is now used. Uncovered by a
test that failed only when the whole suite ran; there is a specific test that spawns two
subprocesses and compares the resulting program.

### D-034 · 2026-09-06 · Re-running a program rewrites its trace
A program is a deterministic function of (case, seed, parameters), so running it again
reproduces the same trace instead of duplicating it. Within a run `events.jsonl` remains
append-only, which is what §10.2 asks for.

### D-035 · 2026-09-06 · Measured bias in the belief about the counterpart's reservation value
In the reference program both parties **overestimate** the other's reservation value (final
medians 0.49 and 0.57 against truths of 0.18 and 0.36), with 90 % coverage 1.0 and 50 % coverage
0–0.5: the belief concentrates (dispersion 0.29 → 0.14) but around a value that is too high. The
cause is structural in f01: it bounds rv below the lowest observed offer, and a counterpart that
never concedes below its target leaves that bound high.
**Pending for Part 3:** measure whether the bias persists when the counterpart reaches its
deadline, and whether f03 corrects it by estimating the concession shape.

### D-036 · 2026-09-06 · What kimi-k3 imposes on the design
Credentials verified against `https://api.moonshot.ai/v1` (models available: `kimi-k3`,
`kimi-k2.6`, `kimi-k2.7-code`, `kimi-k2.7-code-highspeed`). Three real constraints, discovered by
testing, not by reading:

1. **The temperature is fixed.** `temperature != 1` returns `400 invalid temperature`. The client
   detects that error once and omits the parameter from then on.
2. **It is a reasoning model.** The reply comes in `content` and the chain of thought in
   `reasoning_content`; `max_tokens` covers **both**. With `max_tokens=64`, the 64 went to
   reasoning and `content` came back empty. The client retries once with double the budget and,
   if still empty, raises `EmptyCompletion` with the diagnosis instead of returning empty text.
3. **It is slow.** A planning call can take minutes. The timeout went from 180 s to 600 s and —
   more importantly — **the read timeout is no longer retried**: retrying speeds nothing up and
   pays for the same work three times. Only 429 and 5xx errors are retried.

**Consequence for reproducibility (§3):** with the temperature fixed at 1 the model's outputs are
not deterministic. The decision core is (foxes and personality), and the trace stores the raw
reply and the reasoning of every call, so a program can be rebuilt from what the model actually
said. The seed controls everything except the text.

### D-037 · 2026-09-06 · The model writes; the foxes and the personality decide
`ScriptedNegotiator` chooses the action; `TurnWriter` only writes the memo's rationale and the
message at the table, and cannot alter the offer. If the model fails or there are no credentials,
the turn continues with the deterministic memo: the negotiation does not stop for a writing
problem. It is the operational form of rules 7 and 8 of the protocol (§6.5) — the LLM cannot
override the foxes — and it makes the experiment interpretable: what changes between programs is
the belief and the policy, not the eloquence of the turn.

### D-038 · 2026-09-06 · The planner's strategy is validated before it is accepted
The LLM planner produces JSON that is rejected if: the reservation-value quote does not appear
literally in its own corpus (fuzzy comparison, threshold 0.82), the value falls outside the case's
space, the plan names a non-existent or `candidate` fox, uses a weight fox in a single-issue
case, or registers a hypothesis without a test. A strategy that does not validate is not used:
the step fails and says so.

In addition, the preparation memo contrasts the reservation value the model extracts with the
one the deterministic route extracts (`agents/declared_utility.py`). Two independent readings of
the same material: if they agree, the number is anchored; if not, review before sealing.

### D-039 · 2026-09-06 · Model calls go behind a CLI with a dry-run
`scripts/run_preparation.py` does not call the model by default: it prints the prompt and an
estimate of input tokens (~3,300 per party on Parker-Gibson) and needs `--go` to execute, with
`--role` to do one party at a time. The cost is visible before it is incurred.

### D-040 · 2026-09-06 · Scope validation is done against the declared conditions, not against `estimates`
The first real strategy kimi-k3 produced was rejected by a rule of mine that was wrong: "if the
fox estimates `theta.w` and the case has a single issue, it is out of scope". That caught **f06**,
the control, which declares that it can produce a flat prior over any parameter — weights
included — but whose scope condition is literally *"always applicable"*.

The model's plan was correct: rule 2 of the protocol (§6.5) requires the control to run in every
program. The check now reads `scope_conditions` and looks for the multi-issue requirement, which
is where f02 and f09 declare it. There is a regression test.

An instructive case for the rest of the project: when the validator and the agent disagree, the
first suspect must be the validator. That is why the raw reply of every call is persisted in
`llm_calls.jsonl`: it allowed the strategy to be re-validated with the corrected rule **without
paying for the call again**.

### D-041 · 2026-09-06 · What the planner produces when it works
First valid strategy (parkers, 3,853 input tokens, 15,921 output of which 10,691 reasoning,
426 s). Worth recording what it got right, because it sets the bar:

- it extracted its reservation value with the exact quote, matching the deterministic
  route;
- it anchored the first offer at 50,000, justifying it by the optimism quantile;
- it planned 9 tasks, each with its scope condition, and **conditioned f01 on verifying it**
  ("will hold only if I verify it: mean of the first third of its offers...");
- it discarded three foxes on scope with correct reasons: f02 and f09 for multi-issue, and f08
  because it *cannot verify* that the counterpart belongs to a simulated family;
- it discarded MESO, logrolling and contingent contracts citing the case's own material ("You
  are only interested in a straight cash deal");
- it registered 5 falsifiable hypotheses, one of them about whether the counterpart knows they
  are moving — i.e. about its own informational exposure.

None of those decisions was taken by the code: the model took them reading the catalog. What the
code did was stop it from skipping scope.

### D-042 · 2026-09-06 · Hypotheses need a structured threshold to be scorable
The first program with model strategies registered 8 hypotheses and resolved **zero**: the
planner wrote them in prose ("its reservation value is at least 15,500 USD") and the feedback had
nothing to compare them with. A hypothesis that cannot be scored is, for the purposes of §9, the
same as not having registered it.

Three changes:
1. The planner prompt requires `threshold: {comparison, values, unit}` on every hypothesis about
   a numeric parameter, and says so explicitly: without that field it cannot be scored.
2. The feedback has a conservative fallback that extracts the threshold from the statement
   (`hypotheses.thresholds`, English and Spanish phrasings, negations handled). If there is no
   clear pattern it returns `None` and the hypothesis stays unresolved: **inventing a threshold in
   order to score would be worse than not scoring**.
3. `unresolved` is no longer terminal. It means "I could not score it with what I had", so an
   improved feedback must be able to retry it; only `supported` and `refuted` are final.

Result on the reference program: parkers 2/3 resolved (Brier 0.325), gibsons 1/5 (Brier 0.250).
The ones left unresolved are behavioural and about `theta.beta`, which **this case's sealed truth
does not fix**: only the reservation values are verifiable here. The feedback says so on each one
instead of forcing a verdict.

### D-043 · 2026-09-06 · Two trace gaps that only appeared with the complete program
- **Hypotheses did not travel** when a preparation was reused: the feedback resolved an empty
  set and claimed, falsely, that the party had registered none. They are now copied with the
  strategy.
- **Fox calls did not reach the trace**: they stayed in each negotiator's in-memory registry.
  Without them the feedback cannot audit where a belief came from (rule 12 of §6.5). They are now
  emitted as `fox_call` events after every round.

### D-044 · 2026-09-06 · Preparation is reused across programs
A preparation costs ~5 minutes and ~15k tokens per party, and **does not depend on the
negotiation seed or on τ or λ**: repeating it to sweep parameters would be throwing money away.
`gym.run --reuse-preparation <program_id>` takes the strategies and hypotheses already produced,
copying them into the new program so it is self-contained (REVIEW F8). It is also what makes a
sweep comparable: the strategy stays fixed and the only thing that varies is the policy.

### D-045 · 2026-09-06 · First complete program with model strategies
`parker_gibson-20260906-1038214a`: preparation with the planner (kimi-k3), deterministic
decision, feedback against the sealed truth.

- Anchors adopted from the plan, both inside the space and above their own reserve (the gym
  validates them before use).
- **Agreement at 27,000 in round 13**, close to the Nash point and on the Pareto frontier.
- Leave-one-out attribution (pre-review numbers, superseded by D-047): f03 contributes and f01
  subtracts. It is the measurable counterpart of the D-035 bias: f01's bound stays high and
  contaminates the pool. First evidence inside a real program that a fox validated on the
  synthetic dataset can subtract in the case, which is exactly what the attribution exists to detect.
- Isolation: 171 canary checks, 0 leaks.

### D-046 · 2026-09-06 · Architecture review against the original intent
The repository was read against the specification's idea — a multi-agent negotiation system
whose every estimate comes from paper-derived, calibrated foxes — and twelve findings were
recorded in `REVIEW.md` (F1–F12). Nine were fixed in code:

- F1 the online foxes now come from the planner's validated plan (`online_foxes_from_plan`);
- F2 hypotheses are updated during the rounds from the bounds the counterpart's offers imply;
- F3 `gym/replay.py` re-derives every recorded decision and rebuilds DuckDB from `events.jsonl`;
- F4 the counterpart-utility estimate is explicit and fails loudly for multi-issue cases;
- F5 foxes receive responses (implied `reject`, `accept`), and `counterpart_offers()` filters
  proposals only;
- F6 the control f06 is idempotent on the same state (v0.1.1);
- F7 the pool and the leave-one-out attribution exclude the control and experimental foxes;
- F8 a reused preparation is self-contained and fox calls are drained to the trace every round;
- F9 the memo records the counterpart-utility model and pool membership; belief rows store
  per-fox and control quantiles.

Three remain, documented: F10 token budgets are not enforced; F11 f03's validation is in-family;
F12 the foxes are the lineage of the papers, not reproductions of published numbers.
**Discarded:** rewriting the negotiator around the LLM (would violate rules 7–8 of §6.5).

### D-047 · 2026-09-06 · Reference program after the review
Re-run of `parker_gibson-20260906-1038214a` with the fixes of D-046, same seed and reused
preparation: still **agreement at 27,000 in round 13**. Replay reproduces 26/26
decisions; DuckDB rebuilt from the trace: 175 events, 26 actions, 112 fox calls.

Corrected leave-one-out attribution (Δ log score of the pooled belief, control excluded):
**f03 +1.356 (gibsons) / +0.628 (parkers); f01 −0.470 / −0.249.** The pre-review numbers of
D-045 were diluted by the flat control sitting inside the pool; the sign is the same, the
magnitude is now meaningful. Hypothesis updates during the rounds: parkers 1, gibsons 0 — the
gibsons' hypotheses are of the form "rv ≤ X" about the seller, which the seller's offers cannot
decide; the trace says so instead of forcing a verdict.

### D-048 · 2026-09-06 · English as the single working language (supersedes D-013)
The owner asked for English everywhere. Code, comments, prompts, generated reports, catalog,
skills, tests and documentation were rewritten; generated artifacts (validation reports,
`kb_report.md`, `review_queue.md`, feedback reports) were regenerated. Two deliberate exceptions:
the strategies and memos the model produced under the Spanish-era prompts in `runs/` are kept as
recorded data (they are what the programs actually ran on), and `hypotheses.thresholds` still
understands Spanish phrasings so those recorded hypotheses remain scorable.

### D-049 · 2026-09-18 · The project's name is `negotiation-foxes`, everywhere
`comparator` was the pre-existing directory name (D-001), never the project's. The published
repository, `pyproject.toml` and the README title already read `negotiation-foxes`; the working
directory now matches, so a single name identifies the project in the shell, in the package
metadata and on GitHub.
**Discarded:** renaming the repository to `foxes-at-the-table` or `zopa` — the first is more
evocative but the public URL is already in use, the second drops the fox framing the whole
README is built on.
**Revert:** `mv ~/repos-new/negotiation-foxes ~/repos-new/comparator`; nothing in the code reads
the absolute path, and the git remote is unaffected.
