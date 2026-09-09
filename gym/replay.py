"""Deterministic replay of a program from its trace (task 3.2, §10.1).

Replay re-derives every decision of a recorded program from the same inputs — case, seed,
personalities, adopted strategies and the *recorded* counterpart actions — and checks that the
deterministic core chooses exactly the action the trace holds. No model is called: with an LLM
in the loop (temperature fixed at 1) the prose is not reproducible, the decisions are, and this
is what proves it.

It also rebuilds the analytical DuckDB tables from `events.jsonl`, which is the source of truth
(§10.2).

Usage:
    python -m gym.replay --program <program_id>            # verify decisions
    python -m gym.replay --program <program_id> --rebuild-db
    python -m gym.replay --rebuild-db                        # all programs under runs/
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from agents.declared_utility import derive_from_view
from agents.negotiator import ScriptedNegotiator
from agents.personality import Personality
from foxes.registry import FoxRegistry
from gym.case import CaseBundle
from gym.protocol import Action, Protocol
from gym.run import online_foxes_from_plan, stable_offset
from gym.trace import rebuild_duckdb
from hypotheses.store import HypothesisStore


@dataclass
class ReplayReport:
    program_id: str
    rounds_checked: int = 0
    actions_checked: int = 0
    mismatches: list[dict] = field(default_factory=list)
    ended_by_recorded: Optional[str] = None
    ended_by_replayed: Optional[str] = None

    @property
    def ok(self) -> bool:
        return not self.mismatches and self.ended_by_recorded == self.ended_by_replayed

    def to_dict(self) -> dict:
        return {"program_id": self.program_id, "ok": self.ok,
                "rounds_checked": self.rounds_checked, "actions_checked": self.actions_checked,
                "mismatches": self.mismatches, "ended_by_recorded": self.ended_by_recorded,
                "ended_by_replayed": self.ended_by_replayed}


def _load_events(runs_dir: Path, program_id: str) -> list[dict]:
    path = runs_dir / program_id / "events.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _same_action(recorded: dict, replayed: Action) -> bool:
    if recorded["action"] != replayed.action:
        return False
    rec_offers = json.dumps(recorded.get("offers") or [], sort_keys=True, default=str)
    rep_offers = json.dumps(replayed.offers or [], sort_keys=True, default=str)
    return rec_offers == rep_offers


def replay_program(program_id: str, runs_dir: Path = Path("runs"),
                   scratch_dir: Optional[Path] = None) -> ReplayReport:
    """Re-derive each recorded action and compare. Hypothesis stores are written to `scratch_dir`
    (or a sibling `replay/` folder) so the replay never touches the recorded program."""
    events = _load_events(runs_dir, program_id)
    config = [e for e in events if e["kind"] == "program_start"][-1]["payload"]["config"]
    end = [e for e in events if e["kind"] == "program_end"][-1]["payload"]
    report = ReplayReport(program_id=program_id, ended_by_recorded=end.get("ended_by"))

    bundle = CaseBundle.load(config["case_id"])
    role_ids = bundle.role_ids()
    views = {rid: bundle.party_view(rid) for rid in role_ids}
    protocol = Protocol(bundle.domain(), views[role_ids[0]].protocol,
                        list(views[role_ids[0]].issues_spec))
    protocol.max_rounds = int(config["max_rounds"])
    base = Personality.load(config.get("personality", "econ"))
    scratch = scratch_dir or (runs_dir / program_id / "replay")

    # Adopted strategies are read back from the recorded preparation, never re-generated.
    # A program recorded before strategies were copied on reuse keeps them in the source
    # program's directory; the `preparation_reused` event says where.
    reused = [e for e in events if e["kind"] == "preparation_reused"]
    source_program = reused[-1]["payload"].get("source") if reused else None
    negotiators: dict[str, ScriptedNegotiator] = {}
    for rid in role_ids:
        personality = base.with_overrides(
            optimism_quantile_tau=config["tau"][rid], info_gain_weight_lambda=config["lambda"][rid])
        strategy = None
        for candidate in (program_id, source_program):
            if candidate is None:
                continue
            strategy_path = runs_dir / candidate / rid / "prep" / "strategy.json"
            if strategy_path.exists():
                strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
                break
        registry = FoxRegistry()
        online = online_foxes_from_plan(strategy, registry) if strategy else None
        store = HypothesisStore(program_id, rid, runs_dir=scratch)
        negotiator = ScriptedNegotiator(
            views[rid], derive_from_view(views[rid]), personality, protocol, store,
            program_id=program_id, registry=registry,
            seed=int(config["seed"]) + stable_offset(rid), online_foxes=online)
        if strategy:
            negotiator.adopt_strategy(strategy)
        negotiator.preparation_artifacts()
        negotiators[rid] = negotiator

    recorded = [e for e in events if e["kind"] == "action"]
    last_action: Optional[Action] = None
    replayed_end = "deadline"
    for event in recorded:
        rid, round_idx = event["party"], int(event["round"])
        other = bundle.counterpart_of(rid)
        offer_on_table = (last_action.offer if last_action and last_action.party == other
                          else None)
        action, _memo = negotiators[rid].decide(round_idx, offer_on_table)
        report.actions_checked += 1
        report.rounds_checked = max(report.rounds_checked, round_idx)
        if not _same_action(event["payload"], action):
            report.mismatches.append({
                "round": round_idx, "party": rid,
                "recorded": {"action": event["payload"]["action"],
                             "offers": event["payload"].get("offers")},
                "replayed": {"action": action.action, "offers": action.offers}})
        # The trace is authoritative: what the counterpart *actually* saw is the recorded
        # action, so that is what feeds the other party — even after a mismatch.
        authoritative = Action(round_idx, rid, event["payload"]["action"],
                               offers=event["payload"].get("offers") or [],
                               message=event["payload"].get("message"))
        for observer in role_ids:
            if observer != rid:
                negotiators[observer].observe(authoritative)
        last_action = authoritative
        if authoritative.action == "accept":
            replayed_end = f"accept:{rid}"
        elif authoritative.action == "walk_away":
            replayed_end = f"walk_away:{rid}"
    report.ended_by_replayed = replayed_end
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay a program from its trace (task 3.2)")
    ap.add_argument("--program", default=None)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--rebuild-db", action="store_true",
                    help="regenerate runs/trace.duckdb from every events.jsonl")
    args = ap.parse_args()
    runs_dir = Path(args.runs_dir)

    if args.program:
        report = replay_program(args.program, runs_dir)
        status = "OK" if report.ok else f"{len(report.mismatches)} MISMATCH(ES)"
        print(f"replay {args.program}: {status} · {report.actions_checked} actions over "
              f"{report.rounds_checked} rounds · ended {report.ended_by_recorded} / "
              f"replayed {report.ended_by_replayed}")
        for m in report.mismatches[:10]:
            print(f"  round {m['round']} {m['party']}: recorded {m['recorded']} vs "
                  f"replayed {m['replayed']}")
        out = runs_dir / args.program / "replay" / "report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
        print(f"-> {out}")

    if args.rebuild_db:
        programs = [args.program] if args.program else sorted(
            p.name for p in runs_dir.glob("*") if (p / "events.jsonl").exists())
        counts = rebuild_duckdb(programs, runs_dir=runs_dir, db_path=runs_dir / "trace.duckdb")
        print(f"duckdb rebuilt from {len(programs)} program(s): {counts}")


if __name__ == "__main__":
    main()
