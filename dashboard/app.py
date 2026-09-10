"""E1 dashboard (task 3.4): the five views of §11.

    streamlit run dashboard/app.py

Reads `runs/` directly. It computes no metrics: it takes them from `feedback/`, which is what
measures them against the sealed truth. Here they are only displayed.

Rights notice: no text from the case material is shown (see `cases/*/NOTICE.md`); parameters,
offers, beliefs and metrics are.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RUNS = Path("runs")


@st.cache_data
def list_programs() -> list[str]:
    return sorted((p.name for p in RUNS.glob("*") if (p / "events.jsonl").exists()), reverse=True)


@st.cache_data
def load_events(program_id: str) -> list[dict]:
    path = RUNS / program_id / "events.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


@st.cache_data
def load_csv(program_id: str, name: str) -> pd.DataFrame:
    path = RUNS / program_id / "feedback" / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_metrics(program_id: str) -> dict:
    path = RUNS / program_id / "feedback" / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def view_program(program_id: str, events: list[dict]) -> None:
    st.header("Program")
    config = [e for e in events if e["kind"] == "program_start"][-1]["payload"]["config"]
    end = [e for e in events if e["kind"] == "program_end"]
    metrics = load_metrics(program_id).get("outcome", {})

    cols = st.columns(4)
    cols[0].metric("Case", config["case_id"])
    cols[1].metric("Agent", config["agent"])
    cols[2].metric("Rounds", metrics.get("rounds", "—"))
    cols[3].metric("Outcome", "agreement" if metrics.get("agreement") else "impasse")

    st.write("**Personality parameters per party**")
    st.json({"τ": config.get("tau"), "λ": config.get("lambda"), "seed": config.get("seed"),
             "horizon": config.get("max_rounds")})
    if end:
        st.write("**Isolation**")
        st.json(end[-1]["payload"].get("isolation", {}))
    st.write("**Validated foxes available**")
    st.write(", ".join(f"`{f}`" for f in config.get("foxes_validated", [])))


def view_preparation(program_id: str, events: list[dict]) -> None:
    st.header("Preparation")
    preps = [e for e in events if e["kind"] == "preparation"]
    if not preps:
        st.info("This program recorded no preparation.")
        return
    for event in preps:
        party = event["party"]
        payload = event["payload"]
        st.subheader(party)
        prep = payload.get("prep", payload)
        declared = prep.get("declared_utility") or payload.get("declared_utility") or {}
        cols = st.columns(3)
        cols[0].metric("Reservation value", f"{declared.get('reservation_value_raw', '—')}")
        cols[1].metric("Aspiration (utility)",
                       f"{prep.get('aspiration_utility', float('nan')):.3f}"
                       if prep.get("aspiration_utility") else "—")
        evpi = prep.get("evpi_rv_by_round") or {}
        cols[2].metric("EVPI at the end of the horizon",
                       f"{list(evpi.values())[-1]:.4f}" if evpi else "—")
        if evpi:
            st.caption("Value of perfect information about the counterpart's reservation value, "
                       "by round: almost nil at the start, growing as the close approaches.")
            st.plotly_chart(px.bar(x=list(evpi), y=list(evpi.values()),
                                   labels={"x": "round", "y": "EVPI"}),
                            use_container_width=True, key=f"evpi-{party}")
        strategy_path = RUNS / program_id / party / "prep" / "strategy.md"
        if strategy_path.exists():
            with st.expander(f"Strategy of {party}"):
                st.markdown(strategy_path.read_text(encoding="utf-8"))


def view_negotiation(program_id: str, events: list[dict]) -> None:
    st.header("Negotiation")
    actions = [e for e in events if e["kind"] == "action"]
    if not actions:
        st.info("No actions recorded.")
        return
    rows = []
    for event in actions:
        offers = event["payload"].get("offers") or [{}]
        issue = next(iter(offers[0]), None)
        rows.append({"round": event["round"], "party": event["party"],
                     "action": event["payload"]["action"],
                     "offer": offers[0].get(issue) if issue else None,
                     "message": event["payload"].get("message")})
    df = pd.DataFrame(rows)
    fig = px.line(df.dropna(subset=["offer"]), x="round", y="offer", color="party",
                  markers=True, title="Offer trajectory")
    metrics = load_metrics(program_id).get("outcome", {})
    if metrics.get("agreed_offer"):
        price = list(metrics["agreed_offer"].values())[0]
        fig.add_hline(y=price, line_dash="dot",
                      annotation_text=f"agreement {price:,.0f}")
    zopa = metrics.get("true_zopa")
    if zopa:
        fig.add_hrect(y0=min(zopa), y1=max(zopa), fillcolor="green", opacity=0.08,
                      annotation_text="true ZOPA (visible only in feedback)")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, use_container_width=True)

    st.subheader("Decision memos")
    party = st.selectbox("Party", sorted(df.party.unique()), key="memo-party")
    memos = sorted((RUNS / program_id / party / "negotiation").glob("memo_r*.json"))
    if memos:
        choice = st.select_slider("Round", options=[m.stem.split("_r")[1] for m in memos],
                                  key="memo-round")
        memo = json.loads(next(m for m in memos if m.stem.endswith(choice))
                          .read_text(encoding="utf-8"))
        st.write(f"**Rationale**: {memo.get('rationale_written') or memo.get('rationale')}")
        if memo.get("message_written"):
            st.write(f"**Message at the table**: {memo['message_written']}")
        st.write(f"**Expects to learn**: {memo.get('expects_to_learn')}")
        st.json(memo, expanded=False)


def view_feedback(program_id: str) -> None:
    st.header("Feedback")
    metrics = load_metrics(program_id).get("outcome", {})
    if not metrics:
        st.info("This program has no computed feedback. "
                f"Run: `python -m feedback.report --program {program_id}`")
        return
    cols = st.columns(4)
    cols[0].metric("Joint utility", f"{metrics.get('joint_utility', 0):.3f}")
    cols[1].metric("Distance to Pareto", f"{metrics.get('dist_to_pareto', 0):.4f}")
    cols[2].metric("Distance to Nash", f"{metrics.get('dist_to_nash', 0):.4f}")
    cols[3].metric("Nash product", f"{metrics.get('nash_product', 0):.4f}")
    st.write("**Surplus split**")
    st.json(metrics.get("surplus_split", {}))

    calib = load_csv(program_id, "calibration.csv")
    if not calib.empty:
        st.subheader("Calibration of the belief about the counterpart")
        fig = go.Figure()
        for party, group in calib.groupby("party"):
            fig.add_trace(go.Scatter(x=group["round"], y=group["median"], mode="lines+markers",
                                     name=f"{party}: estimated median"))
            fig.add_hline(y=group["true_rv"].iloc[0], line_dash="dash",
                          annotation_text=f"truth of {party}'s counterpart")
        fig.update_layout(title="Estimate of the counterpart's reservation value, round by round",
                          xaxis_title="round", yaxis_title="rv (counterpart's utility scale)")
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(px.line(calib, x="round", y="sd", color="party", markers=True,
                                title="Belief dispersion (concentration by round)"),
                        use_container_width=True)
        st.dataframe(calib.groupby("party")[["abs_error", "crps", "log_score", "cover50",
                                             "cover90"]].mean().round(4),
                     use_container_width=True)

    attrib = load_csv(program_id, "attribution.csv")
    if not attrib.empty:
        st.subheader("Per-fox attribution (leave-one-out)")
        summary = attrib.groupby(["party", "fox_id"])["delta_log_score"].mean().reset_index()
        st.plotly_chart(px.bar(summary, x="fox_id", y="delta_log_score", color="party",
                               barmode="group",
                               title="Contribution to the log score of the pooled belief"),
                        use_container_width=True)

    report = RUNS / program_id / "feedback" / "report.md"
    if report.exists():
        with st.expander("Full debrief"):
            st.markdown(report.read_text(encoding="utf-8"))


def view_batch() -> None:
    st.header("Batches")
    candidates = sorted(RUNS.glob("batch*"))
    if not candidates:
        st.info("No batches. Run: `python -m gym.batch --seeds 5 --sweep --focal <role>`")
        return
    choice = st.selectbox("Batch", [p.name for p in candidates])
    base = RUNS / choice
    programs = pd.read_csv(base / "programs.csv") if (base / "programs.csv").exists() \
        else pd.DataFrame()
    front = pd.read_csv(base / "frontier.csv") if (base / "frontier.csv").exists() \
        else pd.DataFrame()
    if front.empty:
        st.info("That batch has no computed frontier.")
        return
    util_col = next(c for c in front.columns if c.startswith("u_") and c.endswith("_mean"))
    st.subheader("Utility–impasse frontier")
    fig = px.scatter(front, x="impasse_rate", y=util_col, color="tau", size="lambda" if
                     front["lambda"].max() > 0 else None, hover_data=["tau", "lambda", "n"],
                     labels={"impasse_rate": "impasse rate", util_col: "mean utility"})
    st.plotly_chart(fig, use_container_width=True)
    if front["impasse_rate"].max() == 0:
        st.warning("The impasse rate is 0 across the whole sweep: in this case the ZOPA is wide "
                   "and the horizon leaves slack, so the impasse arm of the frontier does not "
                   "appear. Tracing it needs a case with a narrow or uncertain ZOPA.")
    st.plotly_chart(px.line(front, x="tau", y=util_col, color="lambda", markers=True,
                            error_y=front[util_col.replace("_mean", "_hi")] - front[util_col],
                            error_y_minus=front[util_col] - front[util_col.replace("_mean", "_lo")],
                            title="Effect of optimism τ (bootstrap intervals)"),
                    use_container_width=True)
    st.dataframe(front.round(4), use_container_width=True)
    if not programs.empty:
        with st.expander("Programs in the batch"):
            st.dataframe(programs, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="E1 — negotiation with foxes", layout="wide")
    st.title("E1 — bilateral negotiation with foxes")
    programs = list_programs()
    if not programs:
        st.warning("No programs in `runs/`. Run: `python -m gym.run --case parker_gibson`")
        return
    program_id = st.sidebar.selectbox("Program", programs)
    view = st.sidebar.radio("View", ["Program", "Preparation", "Negotiation", "Feedback",
                                     "Batches"])
    events = load_events(program_id)
    if view == "Program":
        view_program(program_id, events)
    elif view == "Preparation":
        view_preparation(program_id, events)
    elif view == "Negotiation":
        view_negotiation(program_id, events)
    elif view == "Feedback":
        view_feedback(program_id)
    else:
        view_batch()


if __name__ == "__main__":
    main()
