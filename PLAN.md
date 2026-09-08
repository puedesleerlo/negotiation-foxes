# PLAN.md — E1: negotiation system with foxes

Execution plan derived from the project owner's specification (sections 1–17).
Status: **Part 1 complete** (Checkpoint 1 delivered, 2026-09-06). **Parts 2 and 3 running end
to end** on the real Parker-Gibson case: full program with Kimi K3 writing the strategies,
deterministic decisions, batch with sweep, feedback against the sealed truth and deterministic
replay. An architecture review against the original intent (REVIEW.md, 2026-09-06) fixed nine
findings and left three documented.
Written 2026-09-05; section 2 now carries the measured result of every task.

---

## 0. Actual state of the inputs (findings, before planning)

The specification assumes `knowledge/kosmos/` with the Kosmos research and `cases/` with at
least one two-sided case. What the repository actually held at the start:

### 0.1 Kosmos research

`knowledge/kosmos/research/Edison Playground.pdf` — **one** report. It contains:

- The three query *prompts* (L1, L2, L3) as sent to Edison/Literature.
- The three full syntheses: L1 (measuring uncertainty about θ), L2 (exploiting the uncertainty),
  L3 (credit assignment in ensembles).
- A **truncated** reference list: the body cites 55 markers, `[1.1]` … `[23.1]` (23 groups ⇒ 23
  formally cited works), but the References section only reaches `[12.3]` and is cut mid-entry.
  Measured by the ingestor: **35 formal entries in 12 groups** (with title, authors, venue,
  DOI/URL and citation context) and **11 groups without an entry** (`[13.1]`–`[23.1]`) that are
  named in the body.

**Consequence for Part 1:** the ingestor extracts two kinds of reference: (a) the formal entries
of the list (`ref_kind: listed`, with DOI/URL and citation context), and (b) the works named in
the body without a formal entry (`ref_kind: inline`, e.g. "Schweinsberg et al. 2023", "Park et
al. 2019"), resolved by search on Crossref/OpenAlex. Measured: 77 cited claims, 12 `listed`
references and 37 `inline` mentions, of which 8 link automatically to a formal entry and **29
are works that exist only as a mention** (13 of them in L1/L2, i.e. within E1's scope). The
design supports re-ingestion if the full export appears, with no work lost.

**Scope consequence:** L3 (credit assignment) is explicitly *out of scope* for E1 (§17:
credit-based weighting = a later experiment). Its references are ingested and annotated but do
not produce foxes. L1 and L2 feed the catalog.

### 0.2 Case

`cases/elmtree_house/source/Raiffa pages 35-43.pdf` — Raiffa, *The Art and Science of
Negotiation*, ch. 3 **"Elmtree House"**, section "Two Parties, One Issue". Scanned PDF (no text
layer; read by vision).

A real, well-documented case, but **it does not meet E1's requirements as it stands**:

| E1 requirement | Elmtree as it comes | Effect |
|---|---|---|
| Two sides with confidential instructions per role | Narrated entirely from the seller (Steve). No instructions for Wilson. | Blocks the buyer's sealed truth |
| Revealed truth on both sides | Seller's RV explicit ($220,000). Buyer's RV **never revealed**: only that he paid $325,000 ⇒ RV_Wilson ≥ 325,000 | Buyer-side calibration admits only a lower bound |
| Multi-issue (weights, logrolling, MESOs, Pareto frontier) | One explicit issue (price). A second *emerges* at the end (donation to the Financial Aid Fund; free repair work, rejected by company policy) | f02, logrolling and the Pareto frontier have nowhere to operate |

What the case **does** provide, valuable and rare:

- Explicit seller truth: RV = $220,000 (moving to Medford), $275,000 to justify Allston;
  open-market BATNA ≈ $125,000; aspiration $350,000.
- A **prior distribution assessed by a human expert** over the buyer's RV (Raiffa's Figure 1):
  quartiles $250,000 / $475,000, right-skewed, support ≈ $100,000–$700,000. An *external
  benchmark* for f05 (prior elicitation from the corpus).
- A real, complete offer trajectory: 125 → 250 → 275 → 290 → 300 (firm) from the buyer;
  600 → 475 → 425 → 400 → 350 from the seller; close at 300 + a 25 donation = $325,000.

**Resolved (D-024):** the owner provided the real two-sided **Parker-Gibson** case (PON), which
replaced the plan to build a second side for Elmtree. Elmtree stays as a validation case for f05.

---

## 1. Cross-cutting conventions

- **Identifiers.** `program_id = <case_id>-<YYYYMMDD>-<hash8>`; `fox_id = fNN_<slug>`;
  `paper_id = <firstauthor><year><slug4>` (e.g. `baarslag2016oppo`); `hypothesis_id = <role>-hNN`.
- **Fox versioning.** SemVer `MAJOR.MINOR.PATCH`; changes with code, calibration data or
  hyperparameters. Every `Posterior` carries `fox_id`, `version`, `program_id`, `party`, `step`, `round`.
- **Seeds.** Every run receives an explicit `seed`; stored in `runs/<program_id>/config.json`.
  Seed offsets per role use SHA-256, never Python's `hash()` (D-033).
- **Language.** English everywhere: documentation, prompts, reports, code, schemas, ids and file
  names (D-048).
- **Statuses.** `candidate` (not implemented, not usable) → `implemented` (code + minimal test)
  → `validated` (recovers a known θ on the synthetic dataset with reported coverage). Only
  `validated` foxes enter the pool in a program; `implemented` ones can run flagged `experimental`,
  outside the pool.
- **Budgets.** Per step and per researcher task, in tokens and seconds, configurable in
  `agents/personalities/*.yaml`. Recorded in `costs`; token budgets are not enforced yet
  (REVIEW F10).

---

## 2. Part 1 — Knowledge base and fox catalog

**Goal.** Turn the Kosmos research into (a) a base of downloaded, queryable papers and (b) a
catalog of implementable microtheories, each with its skill, its calibration and its validation
report.

| # | Task | Deliverable | Depends on | Est. | Acceptance criterion |
|---|---|---|---|---|---|
| 1.1 | Kosmos ingestion | `knowledge/db/references.jsonl`, `kosmos_claims.jsonl` | — | 2 h | Every reference (listed + inline) has a row with `ref_kind`, citation context and the claim it supports; count reported |
| 1.2 | Metadata resolution | `references.jsonl` enriched (DOI, authors, year, venue, OA status) | 1.1 | 2 h | ≥ 80 % of references resolved against Crossref/OpenAlex; ambiguities flagged `needs_review` |
| 1.3 | PDF download | `knowledge/papers/<paper_id>.pdf` + hash | 1.2 | 2 h | Legitimate routes only (arXiv, publisher OA, repositories). Unavailable ones stay `missing` in a manual queue. Proportion reported |
| 1.4 | Database + extraction | `knowledge/db/papers.sqlite`, `knowledge/extracted/<paper_id>.md` | 1.3 | 3 h | Tables `papers, authors, refs, kosmos_claims, models, datasets, files, annotations` populated; section-wise text for every downloaded PDF |
| 1.5 | Structured annotation | rows in `annotations` + `models` | 1.4 | 3 h | Per paper: which model it proposes, which θ component it estimates, inputs, assumptions, evaluation, code/data, relevance (prep/online), relation to RQ1/RQ2 |
| 1.6 | Implementability rubric | `catalog/rubric.md` + scores | 1.5 | 1 h | 7 criteria × 0–3, documented threshold, justified shortlist |
| 1.7 | Catalog | `catalog/foxes.yaml`, `catalog/tools.yaml` | 1.6 | 2 h | ≥ 6 foxes with status; explicit reconciliation with the seed catalog §6.3 (keep / discard / add + reason) |
| 1.8 | Skills | `skills/foxes/<fox_id>/`, `skills/tools/<tool_id>/` | 1.7 | 3 h | Each skill with `SKILL.md` (8 sections), `scripts/`, `references/`, `assets/` and a passing minimal test |
| 1.9 | Runtime + synthetic dataset | `foxes/base.py`, `registry.py`, `pooling.py`; `data/synthetic/*.parquet` | 1.7 | 4 h | Interface §6.2 implemented; 2,000 episodes/domain × 3 domains with Boulware/Conceder/TFT/Hardliner families and sampled θ; column dictionary |
| 1.10 | Implementation + validation | `foxes/<fox_id>/`, `validation_report.md` | 1.9 | 6 h | ≥ 4 `validated` foxes (f01–f04 or replacements) + f06 + f07; each recovers a known θ with 50/90 % interval coverage reported by family and by number of observations |

**Acceptance of Part 1 (Checkpoint 1) — measured result.**

| Criterion | Result |
|---|---|
| Every Kosmos reference with a row and a status | ✅ 49 references (24 distinct works), 77 cited claims |
| Proportion of PDFs downloaded, with a queue of missing ones | ✅ 14/24 works; 23 entries in `knowledge/db/review_queue.md` |
| Catalog with ≥ 6 foxes, ≥ 4 `validated` | ✅ 10 foxes: 6 `validated`, 2 `implemented`, 2 `candidate` |
| Each skill passes its minimal test | ✅ `foxes/tests` green |
| One command reproduces the reports | ✅ `bash scripts/run_part1.sh` |

**Deviations from the plan, with their reason:**

- **f04 did not reach `validated`** (D-020). Not an implementation failure but a finding: the
  acceptance boundary is not identified within a single session of this protocol.
- **f03 was validated for `beta` only** (D-019). The deadline `T` is not recovered from the offer
  curve, exactly as the survey warns. Replacing the bootstrap with a grid posterior turned an
  overconfident fox (coverage 0.12) into an honest one (0.66/0.80). Its validation is in-family
  (REVIEW F11).
- **Two foxes were added to the seed catalog** (D-021): f09, because rule 2 of the protocol
  requires two estimators per parameter with independent mechanisms, and f10 as `candidate`.
- **The KB's GraphRAG was postponed** (D-016): an FTS5 index over the extracted sections instead.
- **f05 stays `candidate`** (D-007): its validation benchmark is not wired.

---

## 3. Part 2 — Agents: planner, researchers, parties, personalities and hypotheses

**Goal.** Each party, in isolation, produces a complete preparation with distributions, outcomes
and hypotheses, and the same agent negotiates following its strategy.

| # | Task | Deliverable | Depends on | Est. | Acceptance criterion |
|---|---|---|---|---|---|
| 2.0 | Two-sided E1 case | `cases/parker_gibson/{case.yaml, public/, roles/*, NOTICE.md}` | — | 4 h | ✅ **done**: real Parker-Gibson case (PON) provided by the owner; `explicit` sealed truth on both sides; single issue, so f02/f09 and t06/t07 declare themselves out of scope |
| 2.1 | Corpus per party + isolation | `gym/case.py`, `gym/isolation.py` | 2.0 | 4 h | ✅ **done**: `PartyView` contains nothing of the counterpart by construction; canary tests including an indirect leak through a file path. GraphRAG per role pending |
| 2.2 | Planner | `agents/planner/`, `agents/prompts/planner_system.md` | 1.7, 2.1 | 5 h | ✅ **done**: produces the three blocks and lists exactly the foxes with their scope justification. **Validated** before acceptance (real quote, value in range, usable foxes, hypotheses with a test and a structured threshold). The plan's online foxes are the ones the negotiator runs (REVIEW F1) |
| 2.3 | Researchers | `agents/researcher/` | 2.2, 1.9 | 3 h | ✅ **done**: one per task, in parallel with a budget, typed output of §8, explicit failure on a `candidate` fox and exclusion from the pool of whatever declares itself out of scope, with a record |
| 2.4 | Preparation orchestration | `gym/prepare.py`, `scripts/run_preparation.py` | 2.2 | 3 h | ✅ **done**: one planner per role in isolation, artifacts of §5.1, sealing, and a CLI with dry-run that shows the cost before incurring it. Preparations are reusable across programs (D-044) and self-contained (REVIEW F8) |
| 2.5 | Personalities | `agents/personalities/econ.yaml`, `agents/personality.py` | — | 2 h | ✅ **done**: τ and λ exposed and sweepable; rule §7.2 implemented; measured that without optimism the agent approaches its own reservation |
| 2.6 | Hypothesis platform | `hypotheses/` | 2.5 | 3 h | ✅ **done**: registry, update during the rounds (REVIEW F2), priority by VOI, resolution with Brier, isolation per party, automatic hypothesis on fox disagreement |
| 2.7 | Negotiator | `agents/negotiator/{scripted,llm_writer}.py` | 2.5, 2.6, 1.10 | 5 h | ✅ **done**: the deterministic core decides and the LLM writer writes. The action is immutable for the writer and the turn survives a model failure. Foxes receive responses, not only proposals (REVIEW F5) |
| 2.8 | Tools | `tools/decision.py`, `tools/integrative.py`, `skills/tools/*` | 1.9 | 4 h | ✅ **done**: all 8, with skill and tests. Those that do not apply to a single-issue case say so instead of producing something empty |
| 2.9 | Model client | `agents/llm.py`, `scripts/check_llm.py` | — | 2 h | ✅ **done** (Kimi K3, OpenAI-compatible endpoint, verified against `api.moonshot.ai`) |

**Acceptance (Checkpoint 2).** Both roles produce `strategy.md`, `plan.json`,
`distributions.parquet`, `outcomes.json`, `hypotheses.jsonl`, `preparation_memo.md` within
budget; every quantitative claim cites a fox or the corpus; the canary passes. ✅ Reached on
`parker_gibson-20260906-1038214a` (D-045).

---

## 4. Part 3 — Gym, feedback and dashboard

| # | Task | Deliverable | Depends on | Est. | Acceptance criterion |
|---|---|---|---|---|---|
| 3.1 | Protocol and gym | `gym/run.py`, `protocol.py`, API run/replay/batch/export | Part 2 | 5 h | ✅ **done**: alternating offers + free text; validates actions, applies the deadline, seals truths |
| 3.2 | Trace | `events.jsonl`, regenerable DuckDB, `gym/replay.py` | 3.1 | 3 h | ✅ **done** (REVIEW F3): deterministic replay re-derives every recorded decision (26/26 on the reference program) and rebuilds DuckDB from the JSONL |
| 3.3 | Feedback | `feedback/*` | 3.2 | 5 h | ✅ **done**: metrics of §5.3 against the sealed truth; leave-one-out attribution that excludes the control (REVIEW F7); debrief written from computed metrics |
| 3.4 | Dashboard | `dashboard/app.py` | 3.3 | 4 h | ✅ **done**: the five views (program, preparation with EVPI per round, offer trajectory with ZOPA, calibration and attribution, batches with the frontier) |
| 3.5 | End to end + batch | `runs/` with 1 program and a batch of 20 | 3.4 | 3 h | ✅ **done** with a limitation: the impasse arm of the utility–impasse frontier does not appear in this case (D-032) |
| 3.6 | Sparring (optional) | `gym/sparring.py` | 3.1 | 2 h | Not started |

---

## 5. Open questions

- **Q1 — Test case.** Resolved: Parker-Gibson (D-024).
- **Q2 — Models and budget.** Resolved: Kimi K3 through `agents/llm.py` (D-025, D-036). Token
  budgets are recorded but not enforced (REVIEW F10).
- **⚠ Q3 — Kosmos export.** The PDF's reference list is truncated at `[12.3]`. If a full
  export exists (Markdown or uncut PDF) it is re-ingested and improves metadata resolution; if
  not, the works named in the body are resolved by search (current state).
- **⚠ Q4 — Multi-issue case.** E1 currently supports single-issue cases only: the
  counterpart-utility estimate from f02/f09 is not wired (REVIEW F4). A multi-issue case would
  make the weight foxes and t06/t07 reachable.

---

## 6. Critical path

```
1.1 → 1.2 → 1.3 → 1.4 → 1.5 → 1.6 → 1.7 ─┬→ 1.8 ────────────┐
                                          └→ 1.9 → 1.10 ─────┴→ CHECKPOINT 1  ✅
2.0 → 2.1 → 2.2 → 2.3 → 2.4 ─┐
2.5 → 2.6 ───────────────────┴→ 2.7 → CHECKPOINT 2  ✅
2.8 ─────────────────────────┘
→ 3.1 → 3.2 → 3.3 → 3.4 → 3.5 → CHECKPOINT 3  ✅ (frontier limitation noted)
```

## 7. Live risks

| Risk | Mitigation | Status |
|---|---|---|
| Truncated references ⇒ poorly grounded catalog | Double extraction (listed + inline) and resolution by search; cheap re-ingestion | Mitigated |
| Foxes not calibrated in the case's domain | Validation on the synthetic dataset by counterpart family before use; narrow declared scope; scope check per fox at run time | Mitigated; in-family caveat for f03 (REVIEW F11) |
| Information leak between parties | Canary per run, per-process views, trace check on every write | Mitigated, tested |
| The LLM overrides the foxes | Rules 7–8 of protocol §6.5: the deterministic core decides, the writer cannot alter the action | Mitigated by construction |
| f01's bias inflates the belief about the counterpart's rv | Measured (D-035, D-047); attribution detects it per program | Open: needs a counterpart that reaches its deadline, or a different likelihood |
| Budgets not enforced | Costs recorded; enforcement is an owner decision | Open (REVIEW F10) |
