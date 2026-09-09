"""Task 1.10 — Validation of the foxes against known θ on the synthetic dataset.

For each fox it reports, per counterpart family and per number of observations: absolute error
of the median, CRPS, log score, 50 % and 90 % interval coverage, and width of the 90 % interval.
The relevant comparison is not against zero error but against the control f06 (uninformed
prior): a fox only contributes if it beats the control's CRPS and log score.

Oracle condition: the foxes receive the *true* utility each offer gives the counterpart. This
isolates the estimator's error from the error of estimating the other side's utility, which is
a separate problem (and what f02/f09 estimate). It is stated as an assumption in every report.

Calibration and validation use disjoint splits of the synthetic set (hash of the episode_id):
no number in a validation report comes from data seen while calibrating.

Usage:  python -m foxes.validate --fox f01_bayes_rv_concession --episodes 300
        python -m foxes.validate --all --episodes 300
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from foxes.base import BeliefState, Observation
from foxes.scoring import brier, reliability_table, score_posterior
from foxes.synthetic import DOMAINS

DATA = Path("data/synthetic")
OBS_POINTS = (2, 4, 8, 999)   # 999 = all the episode's observations


def bucket_label(k: int) -> str:
    return "all" if k >= 999 else f"{k}"


def split_of(episode_id: str, calibration_fraction: float = 0.4) -> str:
    """Deterministic partition per episode: calibration and validation never mix."""
    h = int(hashlib.sha256(episode_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "calibration" if h < calibration_fraction else "validation"


def load(sample: int, seed: int = 0, split: str = "validation") -> tuple[pd.DataFrame, pd.DataFrame]:
    episodes = pd.read_parquet(DATA / "episodes.parquet")
    moves = pd.read_parquet(DATA / "moves.parquet")
    if split:
        episodes = episodes[episodes.episode_id.map(split_of) == split]
    if sample and sample < len(episodes):
        episodes = episodes.sample(sample, random_state=seed).reset_index(drop=True)
    moves = moves[moves.episode_id.isin(set(episodes.episode_id))]
    return episodes, moves


def counterpart_observations(moves_ep: pd.DataFrame, max_rounds: int = 24) -> list[Observation]:
    out: list[Observation] = []
    for _, row in moves_ep[moves_ep.party == "counterpart"].iterrows():
        out.append(Observation(
            round=int(row["round"]), party="counterpart",
            offer=json.loads(row["offer"]), action="propose",
            utility_to_observer=float(row["u_protagonist"]),
            est_utility_to_proposer=float(row["u_counterpart"]),  # oracle condition
            max_rounds=max_rounds,
        ))
    return out


def _state(fox, domain, obs: list[Observation]) -> BeliefState:
    state = fox.init_state(domain)
    state.data["counterpart_party"] = "counterpart"
    for o in obs:
        state = fox.update(state, o)
    return state


def validate_scalar(fox, param: str, truth_col: str, episodes: pd.DataFrame,
                    moves: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, ep in episodes.iterrows():
        domain = DOMAINS[ep.domain_id]
        obs_all = counterpart_observations(moves[moves.episode_id == ep.episode_id])
        if not obs_all:
            continue
        for k in OBS_POINTS:
            obs = obs_all[:k]
            if len(obs) < min(k, 1):
                continue
            state = _state(fox, domain, obs)
            post = fox.posterior(state)
            if param not in post.params:
                continue
            score = score_posterior(post, param, float(ep[truth_col]))
            rows.append({"episode_id": ep.episode_id, "domain_id": ep.domain_id,
                         "family": ep.cp_family, "n_obs": bucket_label(k),
                         "n_obs_real": len(obs), "scope_ok": post.scope_ok, **score})
    return pd.DataFrame(rows)


def validate_weights(fox, episodes: pd.DataFrame, moves: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, ep in episodes.iterrows():
        domain = DOMAINS[ep.domain_id]
        if len(domain.issues) < 2:
            continue
        truth = np.array(json.loads(ep.cp_weights), dtype=float)
        obs_all = counterpart_observations(moves[moves.episode_id == ep.episode_id])
        for k in OBS_POINTS:
            obs = obs_all[:k]
            if len(obs) < 2:
                continue
            state = _state(fox, domain, obs)
            post = fox.posterior(state)
            for j, issue_id in enumerate(domain.issue_ids):
                param = f"w.{issue_id}"
                if param not in post.params:
                    continue
                score = score_posterior(post, param, float(truth[j]))
                rows.append({"episode_id": ep.episode_id, "domain_id": ep.domain_id,
                             "family": ep.cp_family, "n_obs": bucket_label(k),
                             "n_obs_real": len(obs), "issue": issue_id,
                             "scope_ok": post.scope_ok, **score})
    return pd.DataFrame(rows)


def validate_accept(fox, episodes: pd.DataFrame, moves: pd.DataFrame,
                    fit_first: int = 4) -> pd.DataFrame:
    """Fit with the first `fit_first` responses and predict the following ones (out of sample)."""
    rows: list[dict] = []
    for _, ep in episodes.iterrows():
        domain = DOMAINS[ep.domain_id]
        mv = moves[(moves.episode_id == ep.episode_id) & (moves.party == "protagonist")]
        if len(mv) < fit_first + 1:
            continue
        state = fox.init_state(domain)
        rows_ep = list(mv.itertuples())
        for row in rows_ep[:fit_first]:
            fox.observe_response(state, float(row.u_counterpart), bool(row.accepted_by_other),
                                 float(row.round) / 24.0)
        post = fox.posterior(state)
        for row in rows_ep[fit_first:]:
            est = fox.p_accept(state, json.loads(row.offer),
                               utility_to_counterpart=float(row.u_counterpart),
                               t=float(row.round) / 24.0)
            rows.append({"episode_id": ep.episode_id, "family": ep.cp_family,
                         "n_obs": fit_first, "p_pred": est.value,
                         "accepted": bool(row.accepted_by_other),
                         "interval_width": est.high - est.low, "scope_ok": post.scope_ok})
    return pd.DataFrame(rows)


def verdict(df: pd.DataFrame, ctl: pd.DataFrame, label: str = "the control f06") -> str:
    """Compare the fox (in scope only) against the control and state the verdict."""
    ok = df[df.scope_ok]
    if ok.empty:
        return "**Verdict: not assessable** — no observation stayed in scope."
    lines = [
        f"- in scope: {len(ok)}/{len(df)} evaluations ({len(ok) / len(df):.0%})",
        f"- CRPS: **{ok.crps.mean():.3f}** vs {ctl.crps.mean():.3f} for {label}",
        f"- log score: **{ok.log_score.mean():.3f}** vs {ctl.log_score.mean():.3f} for {label}",
        f"- coverage 50/90: {ok.cover50.mean():.2f} / {ok.cover90.mean():.2f} (nominal 0.50 / 0.90)",
    ]
    better = ok.crps.mean() < ctl.crps.mean() and ok.log_score.mean() > ctl.log_score.mean()
    if better:
        lines.append("- **Verdict: beats the control** under the declared conditions.")
    else:
        lines.append("- **Verdict: does not beat the control in aggregate**; see the breakdown "
                     "by family, which is where the scope of use is decided.")
    return "\n".join(lines)


def by_family_verdict(df: pd.DataFrame, ctl: pd.DataFrame) -> str:
    """Families where the fox beats the control, to write the scope with evidence."""
    ok = df[df.scope_ok]
    if ok.empty:
        return ""
    base_crps, base_ls = ctl.crps.mean(), ctl.log_score.mean()
    rows = []
    for family, g in ok.groupby("family"):
        rows.append({"family": family, "n": len(g), "crps": round(g.crps.mean(), 3),
                     "log_score": round(g.log_score.mean(), 3),
                     "beats_control": bool(g.crps.mean() < base_crps and g.log_score.mean() > base_ls)})
    return pd.DataFrame(rows).to_markdown(index=False)


def aggregate(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    agg = df.groupby(by).agg(
        n=("abs_error", "size"),
        mae=("abs_error", "mean"),
        crps=("crps", "mean"),
        log_score=("log_score", "mean"),
        cover50=("cover50", "mean"),
        cover90=("cover90", "mean"),
        width90=("width90", "mean"),
    ).round(3)
    return agg


def write_report(fox_id: str, title: str, sections: list[tuple[str, str]],
                 header: str, out_path: Path) -> None:
    lines = [f"# Validation report — {fox_id}", "",
             f"Generated by `python -m foxes.validate` on {date.today().isoformat()}.", "",
             header, ""]
    for name, body in sections:
        lines += [f"## {name}", "", body, ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {out_path}")


HEADER_COMMON = """\
**Test condition.** The foxes receive the true utility each offer gives the counterpart (the
*oracle* condition). This isolates the estimator's error from the error of estimating the other
side's utility; in a real program that second error adds on top and comes from f02/f09.

**How to read the table.** `mae` is the absolute error of the median; `crps` and `log_score`
are proper scoring rules (lower CRPS is better, higher log score is better); `cover50` and
`cover90` should approach 0.50 and 0.90 — below means overconfidence, above means uselessly wide
intervals. The control f06 row is the reference: a fox that does not beat it adds no information.

**In-family caveat.** The synthetic counterparts are Faratin–Sierra–Jennings tactics; a fox that
fits that family (f03) is validated in-family here and its numbers are optimistic. Out-of-family
evidence comes from real programs (REVIEW.md, F11)."""


def run_f01(episodes, moves) -> None:
    from foxes.f01_bayes_rv_concession import BayesRvConcession
    from foxes.f06_uninformed_prior import UninformedPrior
    fox, control = BayesRvConcession(), UninformedPrior(params=["rv"])
    df = validate_scalar(fox, "rv", "cp_rv", episodes, moves)
    ctl = validate_scalar(control, "rv", "cp_rv", episodes, moves)
    in_scope, out_scope = df[df.scope_ok], df[~df.scope_ok]
    sections = [
        ("Verdict", verdict(df, ctl)),
        ("In scope, by number of observations", aggregate(in_scope, ["n_obs"]).to_markdown()),
        ("In scope, by family (all observations)",
         aggregate(in_scope[in_scope.n_obs == "all"], ["family"]).to_markdown()),
        ("In which families does it beat the control?", by_family_verdict(df, ctl)),
        ("Out of scope (what the scope condition filters out)",
         aggregate(out_scope, ["family"]).to_markdown() if not out_scope.empty
         else "No evaluation fell out of scope."),
        ("Control f06 (uninformed prior)", aggregate(ctl, ["n_obs"]).to_markdown()),
    ]
    write_report("f01_bayes_rv_concession", "", sections, HEADER_COMMON,
                 Path("foxes/f01_bayes_rv_concession/validation_report.md"))
    df.to_csv("foxes/f01_bayes_rv_concession/validation_raw.csv", index=False)


def run_weights(fox_id: str) -> None:
    from foxes.f02_concession_issue_weights import ConcessionIssueWeights
    from foxes.f06_uninformed_prior import UninformedPrior
    from foxes.f09_hypothesis_issue_weights import HypothesisIssueWeights
    episodes, moves = load(RUN_ARGS["episodes"], split=RUN_ARGS.get("split", "validation"))
    episodes = episodes[episodes.domain_id != "d1_price"]
    fox = ConcessionIssueWeights() if fox_id.startswith("f02") else HypothesisIssueWeights()
    df = validate_weights(fox, episodes, moves)
    ctl_rows = []
    for domain_id, sub in episodes.groupby("domain_id"):
        domain = DOMAINS[domain_id]
        control = UninformedPrior(params=[f"w.{i}" for i in domain.issue_ids])
        ctl_rows.append(validate_weights(control, sub, moves))
    ctl = pd.concat(ctl_rows) if ctl_rows else pd.DataFrame()
    in_scope = df[df.scope_ok]
    sections = [
        ("Verdict", verdict(df, ctl, "the control f06 (flat Dirichlet)")),
        ("In scope, by number of observations", aggregate(in_scope, ["n_obs"]).to_markdown()),
        ("In scope, by domain (all observations)",
         aggregate(in_scope[in_scope.n_obs == "all"], ["domain_id"]).to_markdown()),
        ("In scope, by family (all observations)",
         aggregate(in_scope[in_scope.n_obs == "all"], ["family"]).to_markdown()),
        ("In which families does it beat the control?", by_family_verdict(df, ctl)),
        ("Control f06 (flat Dirichlet)", aggregate(ctl, ["n_obs"]).to_markdown()),
    ]
    write_report(fox_id, "", sections, HEADER_COMMON,
                 Path(f"foxes/{fox_id}/validation_report.md"))
    df.to_csv(f"foxes/{fox_id}/validation_raw.csv", index=False)


def run_f03(episodes, moves) -> None:
    from foxes.f03_time_concession_regression import TimeConcessionRegression
    from foxes.f06_uninformed_prior import UninformedPrior
    fox = TimeConcessionRegression(n_bootstrap=60)
    truth_cols = {"rv": "cp_rv", "beta": "cp_beta", "T": "cp_deadline"}
    rows: dict[str, list] = {p: [] for p in truth_cols}
    for _, ep in episodes.iterrows():
        domain = DOMAINS[ep.domain_id]
        obs_all = counterpart_observations(moves[moves.episode_id == ep.episode_id])
        for k in OBS_POINTS:
            obs = obs_all[:k]
            if len(obs) < 2:
                continue
            state = _state(fox, domain, obs)
            post = fox.posterior(state)          # once, for the three parameters
            for param, col in truth_cols.items():
                rows[param].append({"episode_id": ep.episode_id, "domain_id": ep.domain_id,
                                    "family": ep.cp_family, "n_obs": bucket_label(k),
                                    "n_obs_real": len(obs), "scope_ok": post.scope_ok,
                                    **score_posterior(post, param, float(ep[col]))})
    frames = {p: pd.DataFrame(r) for p, r in rows.items()}
    ctl = {p: validate_scalar(UninformedPrior(params=[p]), p, c, episodes, moves)
           for p, c in (("rv", "cp_rv"), ("beta", "cp_beta"), ("T", "cp_deadline"))}
    sections = []
    for param, df in frames.items():
        ok = df[df.scope_ok]
        sections.append((f"Parameter `{param}` — verdict", verdict(df, ctl[param])))
        sections.append((f"Parameter `{param}` in scope, by number of observations",
                         aggregate(ok, ["n_obs"]).to_markdown()))
        sections.append((f"Parameter `{param}` in scope, by family (all)",
                         aggregate(ok[ok.n_obs == "all"], ["family"]).to_markdown()))
        sections.append((f"Parameter `{param}`: in which families does it beat the control?",
                         by_family_verdict(df, ctl[param])))
    write_report("f03_time_concession_regression", "", sections, HEADER_COMMON,
                 Path("foxes/f03_time_concession_regression/validation_report.md"))
    for param, df in frames.items():
        df.to_csv(f"foxes/f03_time_concession_regression/validation_raw_{param}.csv", index=False)


def run_f04(episodes, moves) -> None:
    from foxes.f04_accept_boundary_kde import AcceptBoundaryKde
    fox = AcceptBoundaryKde()
    df = validate_accept(fox, episodes, moves)
    if df.empty:
        print("f04: not enough data")
        return
    ok = df[df.scope_ok]
    fell_back = ok.empty
    if fell_back:
        # Do not silence this: if no prediction meets the scope condition, the number below
        # is NOT the fox's performance within its declared scope.
        ok = df
    overall = brier(ok.p_pred.values, ok.accepted.values)
    base_rate = float(ok.accepted.mean())
    climatology = brier(np.full(len(ok), base_rate), ok.accepted.values)
    rel = pd.DataFrame(reliability_table(ok.p_pred.values, ok.accepted.values))
    by_fam = ok.groupby("family").apply(
        lambda g: pd.Series({"n": len(g), "brier": brier(g.p_pred.values, g.accepted.values),
                             "acceptance_rate": g.accepted.mean(),
                             "interval_width": g.interval_width.mean()}),
        include_groups=False).round(3)
    scope_line = (
        f"- **none** of the {len(df)} predictions meets the scope condition (>= 4 responses "
        f"*with variation* are needed; in this protocol the counterpart rejects the first 4 "
        f"almost always). The figures below are therefore the fox operating OUTSIDE its "
        f"declared scope.\n"
        if fell_back else
        f"- out-of-sample predictions in scope: {len(ok)} of {len(df)}\n")
    body = (scope_line +
            f"- **Brier: {overall:.3f}** (climatology, always predicting the base rate "
            f"{base_rate:.3f}: {climatology:.3f})\n"
            f"- Brier skill score against climatology: {1 - overall / climatology:.3f}")
    write_report("f04_accept_boundary_kde", "", [
        ("Summary", body),
        ("Reliability (predicted vs observed)", rel.to_markdown(index=False)),
        ("By counterpart family", by_fam.to_markdown()),
    ], """\
**Test condition.** The boundary is fitted with the counterpart's first 4 responses and all the
following ones of the same episode are predicted (out of sample). The utility of each offer to
the counterpart is the true one (*oracle* condition).

**How to read.** The Brier is compared against climatology (always predicting the base
acceptance rate). A positive skill score means the estimated boundary adds over knowing nothing;
the reliability table shows whether the probabilities are calibrated or merely ordered.""",
                 Path("foxes/f04_accept_boundary_kde/validation_report.md"))
    df.to_csv("foxes/f04_accept_boundary_kde/validation_raw.csv", index=False)


def run_f06(episodes, moves) -> None:
    from foxes.f06_uninformed_prior import UninformedPrior
    sections = []
    for param, col in (("rv", "cp_rv"), ("beta", "cp_beta"), ("T", "cp_deadline")):
        df = validate_scalar(UninformedPrior(params=[param]), param, col, episodes, moves)
        sections.append((f"Parameter `{param}`", aggregate(df, ["n_obs"]).to_markdown()))
    write_report("f06_uninformed_prior", "", sections, """\
**What is validated in a control.** Not accuracy — a flat prior does not hit — but that its
coverage is nominal and that its scores are the ones to beat. If the 90 % coverage is not close
to 0.90, the declared support does not contain θ and every other fox inherits that bias.""",
                 Path("foxes/f06_uninformed_prior/validation_report.md"))


def run_f07(episodes, moves) -> None:
    """f07 is validated as a *propagator*: if the input belief is calibrated its output must be
    too; and with exact θ its output must coincide with the truth."""
    import json as _json
    from foxes.domain import Utility
    from foxes.f07_zopa_pareto_estimator import ZopaParetoEstimator
    from foxes.f07_zopa_pareto_estimator.fox import utility_from_theta

    fox = ZopaParetoEstimator(n_draws=60)
    rng = np.random.default_rng(7)
    exact_rows, prop_rows = [], []
    for _, ep in episodes.head(120).iterrows():
        domain = DOMAINS[ep.domain_id]
        space = domain.outcome_space()
        w_true = np.array(_json.loads(ep.cp_weights), dtype=float)
        vm_true = _json.loads(ep.cp_value_maps)
        # true directions: +1 if the best value sits at the high end of the issue
        directions = []
        for issue in domain.issues:
            vm = vm_true[issue.issue_id]
            if isinstance(vm, list):      # continuous: [worst, best]
                directions.append(1.0 if float(vm[1]) > float(vm[0]) else -1.0)
            else:
                values = list(issue.values)
                directions.append(1.0 if float(vm[values[-1]]) >= float(vm[values[0]]) else -1.0)
        directions = np.array(directions)

        # the real own utility is rebuilt from its own value maps
        own_vm = _json.loads(ep.prot_value_maps)
        own = Utility(domain, dict(zip(domain.issue_ids, _json.loads(ep.prot_weights))),
                      {k: (tuple(v) if isinstance(v, list) else v) for k, v in own_vm.items()})
        own.reservation_value = float(ep.prot_rv)
        other_true = utility_from_theta(domain, w_true, directions, float(ep.cp_rv))
        true_zopa = float(np.mean([(own(o) >= own.reservation_value)
                                   and (other_true(o) >= other_true.reservation_value)
                                   for o in space]))

        # (a) exact θ: propagation must reproduce the truth
        state = fox.init_state(domain, own)
        state.data["theta_samples"] = {"weights": np.tile(w_true, (5, 1)),
                                       "directions": np.tile(directions, (5, 1)),
                                       "rv": np.full(5, float(ep.cp_rv))}
        post = fox.posterior(state)
        exact_rows.append({"episode_id": ep.episode_id,
                           "error_zopa": abs(post.mean("zopa_fraction") - true_zopa)})

        # (b) noisy but calibrated belief: coverage of the true value
        n = 60
        noisy_w = rng.dirichlet(np.clip(w_true * 25, 0.05, None), n)
        noisy_dir = np.where(rng.random((n, len(directions))) < 0.85, directions, -directions)
        noisy_rv = np.clip(rng.normal(float(ep.cp_rv), 0.08, n), 0, 1)
        state = fox.init_state(domain, own)
        state.data["theta_samples"] = {"weights": noisy_w, "directions": noisy_dir, "rv": noisy_rv}
        post = fox.posterior(state)
        prop_rows.append({"episode_id": ep.episode_id, "domain_id": ep.domain_id,
                          "family": ep.cp_family, "n_obs": "noisy belief",
                          "scope_ok": post.scope_ok,
                          **score_posterior(post, "zopa_fraction", true_zopa)})

    exact = pd.DataFrame(exact_rows)
    prop = pd.DataFrame(prop_rows)
    body_exact = (f"- episodes: {len(exact)}\n"
                  f"- mean absolute error with exact θ: **{exact.error_zopa.mean():.4f}**\n"
                  f"- maximum error: {exact.error_zopa.max():.4f}\n\n"
                  "With exact θ the propagation has no degrees of freedom: any error above the "
                  "sampling noise of the outcome space would be an implementation defect.")
    write_report("f07_zopa_pareto_estimator", "", [
        ("Correctness: propagation with exact θ", body_exact),
        ("Calibration: propagation of a noisy belief",
         aggregate(prop, ["domain_id"]).to_markdown()),
        ("By counterpart family", aggregate(prop, ["family"]).to_markdown()),
    ], """\
**What is validated.** f07 does not estimate θ: it *propagates* it. Its validation therefore has
two parts. (a) Correctness: with exact θ the estimated ZOPA must coincide with the true one.
(b) Calibration of the propagation: fed a noisy belief centred on the truth (Dirichlet of
concentration 25 over the weights, 15 % of directions flipped, rv with 0.08 noise), the coverage
of its intervals over the true ZOPA must be nominal.

If (a) fails, there is an implementation error. If (b) fails, f07 distorts the uncertainty it
receives, and every decision that relies on its output inherits that distortion.""",
                 Path("foxes/f07_zopa_pareto_estimator/validation_report.md"))
    prop.to_csv("foxes/f07_zopa_pareto_estimator/validation_raw.csv", index=False)


RUN_ARGS: dict = {"episodes": 300}


def main() -> None:
    ap = argparse.ArgumentParser(description="Fox validation (task 1.10)")
    ap.add_argument("--fox", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--split", default="validation", choices=["validation", "calibration", ""])
    args = ap.parse_args()
    RUN_ARGS["episodes"] = args.episodes
    RUN_ARGS["split"] = args.split

    episodes, moves = load(args.episodes, split=args.split)
    targets = ([args.fox] if args.fox else
               ["f01_bayes_rv_concession", "f02_concession_issue_weights",
                "f03_time_concession_regression", "f04_accept_boundary_kde",
                "f06_uninformed_prior", "f07_zopa_pareto_estimator",
                "f09_hypothesis_issue_weights"])
    for fox_id in targets:
        print(f"\n=== {fox_id} ===")
        if fox_id.startswith("f01"):
            run_f01(episodes, moves)
        elif fox_id.startswith(("f02", "f09")):
            run_weights(fox_id)
        elif fox_id.startswith("f03"):
            run_f03(episodes, moves)
        elif fox_id.startswith("f04"):
            run_f04(episodes, moves)
        elif fox_id.startswith("f06"):
            run_f06(episodes, moves)
        elif fox_id.startswith("f07"):
            run_f07(episodes, moves)
        else:
            print(f"no validation routine for {fox_id}")


if __name__ == "__main__":
    main()
