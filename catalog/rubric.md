# Implementability rubric (task 1.6)

Every candidate model in the corpus is scored 0–3 on seven criteria (maximum 21). The score is
assigned by the annotator with the evidence of `catalog/paper_annotations.yaml` in view, and it
is reviewable: every cell can be argued against the annotation of the corresponding paper.

## Criteria

| # | Criterion | 0 | 1 | 2 | 3 |
|---|---|---|---|---|---|
| C1 | **Complete specification** | Described in prose only | Clear idea, details missing | Formula reconstructible | Explicit, unambiguous formula |
| C2 | **Data available in our environment** | Needs data we neither have nor can generate | Needs external human data | Can be calibrated on the synthetic dataset with assumptions | Calibrates on our synthetic dataset directly |
| C3 | **Compute cost** | Heavy training (RL, GPU) | Expensive fit per round | Seconds per round | Milliseconds per round |
| C4 | **Code available** | No | Description without code | Partial third-party code | Trivial to reimplement or direct code |
| C5 | **Maturity / replication** | Single proposal without evaluation | Evaluated by its authors | Evaluated on a common benchmark | Replicated or a field standard |
| C6 | **Fit to θ** | Estimates nothing of θ | Estimates something tangential | Estimates a component of θ | Estimates a central component (rv, w, beta/T, p_accept) |
| C7 | **Fit to the protocol** | Another protocol (auctions, multi-party) | Adaptable with strong assumptions | Adaptable with mild assumptions | Direct bilateral alternating offers |

## Threshold

- **≥ 15/21 → shortlist**: implemented and validated in Part 1.
- **11–14 → `candidate`**: enters the documented catalog, not implemented in E1.
- **≤ 10 → discarded**: the reason is recorded and it does not enter the catalog.

A model with C2 = 0 cannot go beyond `candidate` whatever its total: without calibration data
there is no validation report, and without a validation report it is not used (the "zero
mock-ups" rule).

## Scores

| Model (paper) | C1 | C2 | C3 | C4 | C5 | C6 | C7 | Total | Decision |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|---|
| Bayesian learning over RV hypotheses (zeng1998baye, via baarslag2015lear §4.1/5.1.1) | 3 | 3 | 3 | 3 | 3 | 3 | 3 | **21** | shortlist → `f01_bayes_rv_concession` |
| Weights by normalised concession ratio (baarslag2015lear §5.3; Niemann and Lang) | 3 | 3 | 3 | 3 | 2 | 3 | 3 | **20** | shortlist → `f02_concession_issue_weights` |
| Non-linear regression on the time-dependent tactic (faratin1998nego; Hou, via baarslag2015lear §3.6/5.1.1) | 3 | 3 | 3 | 2 | 3 | 3 | 3 | **20** | shortlist → `f03_time_concession_regression` |
| Acceptance boundary by KDE (baarslag2015lear §4.3/5.1.2) | 2 | 3 | 3 | 3 | 2 | 3 | 3 | **19** | shortlist → `f04_accept_boundary_kde` |
| Bayes over a hypothesis space of weights + evaluation shapes (konishi2026pref §3.1) | 3 | 3 | 2 | 2 | 1 | 3 | 2 | **16** | shortlist → `f09_hypothesis_issue_weights` |
| Monte Carlo propagation of θ to ZOPA/frontier (banks2020adve, ARA pattern) | 2 | 3 | 3 | 3 | 2 | 2 | 3 | **18** | shortlist → `f07_zopa_pareto_estimator` |
| Outcome simulation with counterpart families (hendrikx2012eval, experimental design) | 2 | 3 | 2 | 2 | 3 | 2 | 3 | **17** | shortlist → `f08_outcome_simulator` |
| Flat prior over θ (control, no paper) | 3 | 3 | 3 | 3 | 3 | 2 | 3 | **20** | shortlist → `f06_uninformed_prior` (mandatory control) |
| Utility–impasse frontier by offer ambition (schweinsberg2022unde) | 2 | 1 | 3 | 0 | 2 | 2 | 2 | **12** | `candidate` → `f10_impasse_risk`: the functional form (three-zone quartic) is reusable, but its calibration is on single-issue eBay; recalibrating it on our synthetic dataset changes the estimated object. Documented and not used in E1 |
| Unscented particle filter over fuzzy preferences (eshragh2019mult) | 1 | 2 | 2 | 0 | 1 | 2 | 2 | **10** | discarded as its own fox: its contribution (sequential particle representation) is already in f01/f09; the fuzzy preference model does not fit additive utility |
| Utility model with Gaussian processes (leahu2019auto) | 2 | 2 | 1 | 0 | 1 | 2 | 2 | **10** | discarded in E1: high cost, and its contribution to RQ2 (value of information) is implemented as the EIG tool, not as a fox |
| Prior elicitation from the corpus with an LLM (no paper; own design, lineage konishi2026pref/lin2026dist) | 2 | 0 | 1 | 3 | 0 | 3 | 3 | **12** | `candidate` → `f05_case_reader_prior`: no validation benchmark wired (DECISIONS D-007). C2 = 0 pins it at `candidate` |
| UAOM, aleatoric/epistemic decomposition in deep RL (yang2025unce) | 1 | 0 | 0 | 1 | 1 | 2 | 1 | **6** | discarded: requires RL training; its useful idea (separating epistemic from aleatoric) is adopted as a metric, not as a model |
| BOND, distilling Bayesian beliefs into an LM (lin2026dist) | 2 | 1 | 1 | 1 | 1 | 2 | 2 | **10** | discarded in E1: requires an LLM and the CaSiNo corpus; its contribution (per-turn Brier on beliefs) is adopted as a calibration metric |
| FortUne Dial, outcome forecasting (sicilia2024deal) | 2 | 1 | 1 | 1 | 1 | 1 | 2 | **9** | discarded in E1: its contribution (Brier + calibration on the outcome) is adopted as a metric |

## Reconciliation with the specification's seed catalog (§6.3)

| Seed | Result | Reason |
|---|---|---|
| `f01_bayes_rv_concession` | **kept** | 21/21; the operational formula is in baarslag2015lear §4.1 and §5.1.1 |
| `f02_freq_issue_weights` | **kept and renamed** to `f02_concession_issue_weights` | The evidence does not describe a frequency count but the normalised concession ratio w_i = 1 − c_i (baarslag2015lear §5.3). The name now says what it does |
| `f03_time_concession_regression` | **kept** | Faratin–Sierra–Jennings with the u(t) form from the survey |
| `f04_accept_boundary_np` | **kept and renamed** to `f04_accept_boundary_kde` | KDE is fixed as the technique (the survey documents it); GP is discarded on cost |
| `f05_case_reader_prior` | **kept as `candidate`** | No validation benchmark wired; not implemented or used |
| `f06_uninformed_prior` | **kept** | Mandatory control of rule 2 of protocol §6.5 |
| `f07_zopa_pareto_estimator` | **kept** | ARA pattern of Monte Carlo propagation |
| `f08_outcome_simulator` | **kept** | Relies on the in-house simulator and the design of hendrikx2012eval |
| — | **added** `f09_hypothesis_issue_weights` | Second estimator of `w` with a different mechanism (Bayesian over hypotheses, not concession ratio): rule 2 of the protocol requires two foxes per parameter with independent methods |
| — | **added** `f10_impasse_risk` (`candidate`) | RQ2 needs the cost side (impasse); the only quantitative evidence in the corpus is from another domain, so it is documented without being used |
