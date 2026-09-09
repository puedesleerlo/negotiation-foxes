"""Run a complete program: preparation, negotiation and (separately) feedback (tasks 2.4, 3.1).

A program is one execution of the three steps on a case, with its own `program_id`, an explicit
seed and a complete trace. The decision core is deterministic; the LLM only writes prose
(`--with-llm`) and/or the preparation strategy (`--llm-preparation`).

Usage:  python -m gym.run --case parker_gibson --seed 7
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from agents.declared_utility import derive_from_view
from agents.negotiator import ScriptedNegotiator
from agents.personality import Personality
from foxes.registry import FoxRegistry
from gym.case import CaseBundle
from gym.isolation import LeakDetector
from gym.protocol import Action, Protocol
from gym.trace import Trace
from hypotheses.store import HypothesisStore


def stable_offset(text: str, modulo: int = 1000) -> int:
    """Seed offset reproducible across processes and machines (Python's `hash()` is not)."""
    return int(hashlib.sha256(text.encode()).hexdigest(), 16) % modulo


def make_program_id(case_id: str, seed: int, tau, lam) -> str:
    stamp = date.today().strftime("%Y%m%d")
    digest = hashlib.sha256(f"{case_id}{seed}{tau}{lam}".encode()).hexdigest()[:8]
    return f"{case_id}-{stamp}-{digest}"


def _personality_for(role_id: str, base: Personality, tau, lam) -> Personality:
    """tau and lambda may be a number (both parties) or a dict per role (asymmetric sweep)."""
    tau_v = tau.get(role_id) if isinstance(tau, dict) else tau
    lam_v = lam.get(role_id) if isinstance(lam, dict) else lam
    if tau_v is None and lam_v is None:
        return base
    return base.with_overrides(
        optimism_quantile_tau=tau_v if tau_v is not None else base.tau,
        info_gain_weight_lambda=lam_v if lam_v is not None else base.lambda_info)


def online_foxes_from_plan(strategy: dict, registry: FoxRegistry) -> list[str]:
    """The foxes the planner scheduled for the online phase — the strategy is the contract.

    Only catalog foxes are kept (tools are called by the negotiator itself). Candidates were
    already refused by the plan validation; anything left is `validated` or `implemented`.
    """
    out: list[str] = []
    for task in strategy.get("plan", []) or []:
        ident = str(task.get("fox_or_tool", ""))
        phase = str(task.get("phase", "")).lower()
        if ident in registry.catalog and phase == "online" and ident not in out:
            out.append(ident)
    return out


def drain_fox_calls(trace: Trace, negotiator: ScriptedNegotiator, party: str) -> int:
    """Emit the fox calls made since the last drain, so `events.jsonl` stays chronological."""
    calls = negotiator.registry.calls
    for call in calls:
        trace.emit("fox_call", party=party, step=call.step, round=call.round,
                   fox_id=call.fox_id, version=call.version, scope_ok=call.scope_ok,
                   warnings=call.warnings, summary=call.summary, seconds=call.seconds,
                   n_observations=call.n_observations, experimental=call.experimental)
    n = len(calls)
    calls.clear()
    return n


def run_program(case_id: str = "parker_gibson", seed: int = 7, tau=None, lam=None,
                runs_dir: Path = Path("runs"), max_rounds: Optional[int] = None,
                verbose: bool = True, with_llm: bool = False,
                llm_preparation: bool = False,
                reuse_preparation_from: Optional[str] = None) -> dict:
    """Two independent uses of the model:

    `llm_preparation=True` — the planner writes each party's strategy before negotiating; the
    negotiator adopts its aspiration, its opening anchor and its list of online foxes, which are
    strategy decisions, not estimates.

    `with_llm=True` — the model writes the memo's reasoning and the table message, but does
    **not** choose the action (see `agents/negotiator/llm_writer.py`).

    `reuse_preparation_from` — take the strategies another program already produced instead of
    calling the model again. Preparation is expensive (minutes and ~15k tokens per party) and
    does not depend on the negotiation seed, so repeating it to sweep tau or lambda would burn
    money for nothing.
    """
    bundle = CaseBundle.load(case_id)
    base = Personality.load("econ")
    personalities = {rid: _personality_for(rid, base, tau, lam) for rid in bundle.role_ids()}
    signature = "|".join(f"{rid}:{p.tau}:{p.lambda_info}" for rid, p in sorted(personalities.items()))
    program_id = make_program_id(case_id, seed, signature, "")

    detector = LeakDetector.for_case(bundle)
    trace = Trace(program_id, root=runs_dir, detector=detector, fresh=True)
    rng = np.random.default_rng(seed)
    role_ids = bundle.role_ids()
    views = {rid: bundle.party_view(rid) for rid in role_ids}
    protocol = Protocol(bundle.domain(), views[role_ids[0]].protocol,
                        list(views[role_ids[0]].issues_spec))
    if max_rounds:
        protocol.max_rounds = max_rounds

    trace.emit("program_start", step="setup", config={
        "case_id": case_id, "seed": seed, "personality": base.personality_id,
        "tau": {rid: p.tau for rid, p in personalities.items()},
        "lambda": {rid: p.lambda_info for rid, p in personalities.items()},
        "max_rounds": protocol.max_rounds,
        "agent": ("ScriptedNegotiator" + (" + TurnWriter(LLM)" if with_llm else "")
                  + (" + PartyPlanner(LLM)" if llm_preparation else "")),
        "foxes_validated": FoxRegistry().available()})

    # ---------------------------------------------------------------- preparation (isolated)
    strategies: dict[str, dict] = {}
    if reuse_preparation_from:
        reused_hypotheses = 0
        for rid in role_ids:
            path = runs_dir / reuse_preparation_from / rid / "prep" / "strategy.json"
            if path.exists():
                strategies[rid] = json.loads(path.read_text(encoding="utf-8"))
                # A program must be self-contained: the replay reads the strategy from the
                # program's own directory, so the reused one is copied in (with its plan).
                for name in ("strategy.json", "plan.json", "strategy.md"):
                    src = runs_dir / reuse_preparation_from / rid / "prep" / name
                    if src.exists():
                        dst = runs_dir / program_id / rid / "prep" / name
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            elif verbose:
                print(f"[prep] {rid}: no strategy to reuse in {reuse_preparation_from}; "
                      f"falling back to the deterministic preparation")
            # Hypotheses are registered in preparation: reusing the strategy must bring them
            # too, or the feedback would resolve an empty set and falsely report that the
            # party registered none.
            source = runs_dir / reuse_preparation_from / rid / "hypotheses.jsonl"
            if source.exists():
                target = runs_dir / program_id / rid / "hypotheses.jsonl"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
                reused_hypotheses += len(source.read_text().strip().splitlines())
        trace.emit("preparation_reused", step="preparation",
                   source=reuse_preparation_from, roles=sorted(strategies),
                   hypotheses_reused=reused_hypotheses)
    elif llm_preparation:
        from gym.prepare import run_preparation
        prepared = run_preparation(case_id, program_id, roles=role_ids, runs_dir=runs_dir)
        for rid, res in prepared.items():
            if res.get("ok"):
                strategies[rid] = res["strategy"]
            else:
                trace.emit("preparation_failed", party=rid, step="preparation",
                           errors=res["validation"]["errors"])
                if verbose:
                    print(f"[prep] {rid}: the strategy did not validate "
                          f"({res['validation']['errors'][0][:90]}); continuing with the "
                          f"deterministic preparation")

    negotiators: dict[str, ScriptedNegotiator] = {}
    for rid in role_ids:
        view = views[rid]
        declared = derive_from_view(view)
        store = HypothesisStore(program_id, rid, runs_dir=runs_dir)
        registry = FoxRegistry()
        online = online_foxes_from_plan(strategies[rid], registry) if rid in strategies else None
        negotiator = ScriptedNegotiator(
            view, declared, personalities[rid], protocol, store, program_id=program_id,
            registry=registry, seed=seed + stable_offset(rid), online_foxes=online)
        negotiators[rid] = negotiator

        if rid in strategies:
            adopted = negotiator.adopt_strategy(strategies[rid])
            trace.emit("strategy_adopted", party=rid, step="preparation", **adopted)
            if verbose:
                print(f"[prep] {rid}: strategy adopted — aspiration "
                      f"{adopted['aspiration_utility']} · anchor {adopted['first_offer']} · "
                      f"online foxes {adopted['online_foxes']}"
                      + (f" · rejected: {adopted['rejected']}" if adopted["rejected"] else ""))
        prep = negotiator.preparation_artifacts()
        drain_fox_calls(trace, negotiator, rid)
        trace.write_json(f"{rid}/prep/declared_utility.json", declared.to_dict(), party=rid)
        trace.write_json(f"{rid}/prep/preparation.json", prep, party=rid)
        trace.emit("preparation", party=rid, step="preparation", prep=prep)
        if verbose:
            evpi_txt = " · ".join(f"r{r}: {v:.4f}" for r, v in prep["evpi_rv_by_round"].items())
            print(f"[prep] {rid}: declared rv {declared.reservation_value_raw:.0f} "
                  f"(utility {declared.reservation_value_utility:.3f}) · "
                  f"aspiration {prep['aspiration_utility']:.3f} · EVPI on rv -> {evpi_txt}")

    # Sealing: from here on the declared utility does not change.
    trace.emit("sealed", step="setup",
               note="declared utilities frozen; the case truth is read by feedback only")

    # ---------------------------------------------------------------- negotiation
    order = [protocol.first_mover(role_ids, rng)]
    order.append(bundle.counterpart_of(order[0]))
    trace.emit("first_mover", step="negotiation", party=order[0])

    writers: dict[str, Any] = {}
    if with_llm:
        from agents.llm import LLMClient
        from agents.negotiator.llm_writer import TurnWriter
        client = LLMClient()
        issue_id = bundle.domain().issue_ids[0]
        for rid in role_ids:
            writers[rid] = TurnWriter(rid, views[rid].side, bundle.counterpart_of(rid),
                                      issue_id, client=client)

    last_message: Optional[str] = None
    last_action: Optional[Action] = None
    outcome = {"agreement": False, "agreed_offer": None, "rounds": 0, "ended_by": "deadline"}
    belief_rows: list[dict] = []
    grid = np.linspace(0.005, 0.995, 200)

    for round_idx in range(1, protocol.max_rounds + 1):
        for rid in order:
            other = bundle.counterpart_of(rid)
            offer_on_table = (last_action.offer if last_action and last_action.party == other
                              else None)
            action, memo = negotiators[rid].decide(round_idx, offer_on_table)
            drain_fox_calls(trace, negotiators[rid], rid)
            check = protocol.validate(action)
            if not check.ok:
                trace.emit("invalid_action", party=rid, step="negotiation", round=round_idx,
                           action=action.action, errors=check.errors)
                action = Action(round_idx, rid, "message_only",
                                message="(invalid action corrected by the gym)")
            if rid in writers:
                written = writers[rid].write(action, memo, last_message=last_message)
                memo["rationale_written"] = written.rationale
                memo["message_written"] = written.message
                memo["disclosure_reasoning"] = written.disclosure_reasoning
                memo["llm_usage"] = written.usage
                memo["llm_reasoning"] = written.reasoning
                if written.concern:
                    memo["llm_concern"] = written.concern
                if written.failed:
                    memo["llm_failed"] = written.failed
                    trace.emit("llm_failed", party=rid, step="negotiation", round=round_idx,
                               reason=written.failed)
                elif written.message and protocol.allow_free_text:
                    action.message = written.message

            trace.write_json(f"{rid}/negotiation/memo_r{round_idx:02d}.json", memo, party=rid)
            trace.emit("decision_memo", party=rid, step="negotiation", round=round_idx,
                       memo=memo)
            trace.emit("action", party=rid, step="negotiation", round=round_idx,
                       action=action.action, offers=action.offers, message=action.message)

            beliefs = negotiators[rid].last_beliefs
            if beliefs is not None:
                sample = np.quantile(beliefs.theta["rv"], grid)
                # Individual posteriors of the foxes that formed the pool — exactly those, so
                # the feedback's leave-one-out attribution is about the pool that decided. The
                # control is kept apart as the reference, never as a pool member.
                per_fox = {fid: [float(post.quantile("rv", q)) for q in grid]
                           for fid, post in beliefs.per_fox.items()
                           if fid in beliefs.in_scope and "rv" in post.params}
                control = beliefs.per_fox.get("f06_uninformed_prior")
                belief_rows.append({
                    "program_id": program_id, "party": rid, "round": round_idx,
                    "param": "rv", "quantile_samples": sample.tolist(),
                    "per_fox_quantiles": json.dumps(per_fox),
                    "control_quantiles": json.dumps(
                        [float(control.quantile("rv", q)) for q in grid] if control else []),
                    "median": float(np.median(beliefs.theta["rv"])),
                    "entropy_proxy": float(np.std(beliefs.theta["rv"])),
                    "foxes_in_scope": ",".join(beliefs.in_scope),
                    "disagreement_sd": beliefs.disagreement_sd})

            for observer in role_ids:
                if observer != rid:
                    negotiators[observer].observe(action)   # foxes + hypotheses, §5.2 steps 2-3
                    drain_fox_calls(trace, negotiators[observer], observer)
            last_action = action
            last_message = action.message
            outcome["rounds"] = round_idx

            if action.action == "accept":
                outcome.update(agreement=True, agreed_offer=offer_on_table,
                               ended_by=f"accept:{rid}")
                break
            if action.action == "walk_away":
                outcome.update(agreement=False, ended_by=f"walk_away:{rid}")
                break
        if outcome["ended_by"] != "deadline":
            break

    for rid in role_ids:
        negotiators[rid].hypotheses.save()
        updated = sum(len(h.updates) for h in negotiators[rid].hypotheses.all())
        trace.emit("hypotheses_saved", party=rid, step="negotiation",
                   n_total=len(negotiators[rid].hypotheses.all()),
                   n_open=len(negotiators[rid].hypotheses.open()),
                   n_updates_during_rounds=updated)
    if belief_rows:
        pd.DataFrame(belief_rows).to_parquet(trace.root / "beliefs.parquet", index=False)

    trace.emit("program_end", step="negotiation", **outcome,
               isolation=detector.summary())
    if verbose:
        print(f"[end] {outcome['ended_by']} in round {outcome['rounds']}; "
              f"agreement: {outcome['agreed_offer']}")
        print(f"[isolation] {detector.summary()}")
    return {"program_id": program_id, "outcome": outcome, "trace": trace,
            "bundle": bundle, "runs_dir": runs_dir, "personalities": personalities}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a complete program (task 3.1)")
    ap.add_argument("--case", default="parker_gibson")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--max-rounds", type=int, default=None)
    ap.add_argument("--with-llm", action="store_true",
                    help="the model writes the memo and the table message (it does not choose the action)")
    ap.add_argument("--llm-preparation", action="store_true",
                    help="the planner writes each party's strategy before negotiating")
    ap.add_argument("--reuse-preparation", default=None, metavar="PROGRAM_ID",
                    help="reuse another program's strategies instead of calling the model again")
    args = ap.parse_args()
    result = run_program(args.case, args.seed, args.tau, args.lam, max_rounds=args.max_rounds,
                         with_llm=args.with_llm, llm_preparation=args.llm_preparation,
                         reuse_preparation_from=args.reuse_preparation)
    print(f"\nprogram: {result['program_id']}")
    print(f"trace:   runs/{result['program_id']}/events.jsonl")


if __name__ == "__main__":
    main()
