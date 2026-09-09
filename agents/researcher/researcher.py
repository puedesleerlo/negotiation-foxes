"""Researchers: ephemeral sub-agents that execute one task of the plan (§8, task 2.3).

One per task. They receive the task, the party's view of the case and its declared utility;
they run the fox the task names and return a **typed** output. They are mechanical on purpose:
their job is to execute and format, not to judge — which is why they need no language model,
as the specification recommends for mechanical tasks.

Constraints (§8), all enforced here rather than trusted to a prompt:
  * they never read the counterpart's corpus — they only receive a `PartyView`, which does not
    contain it;
  * they never invent data — if the fox cannot run they return a failure, not an estimate;
  * a task that requires a `candidate` fox returns an **explicit failure**;
  * they run in parallel with a per-task budget.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from foxes.base import BeliefState, Observation
from foxes.domain import Utility
from foxes.registry import FoxRegistry
from gym.case import PartyView


@dataclass
class ResearchOutput:
    """Exactly the shape §8 asks for."""

    task_id: str
    fox_id: Optional[str]
    version: Optional[str]
    result: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    cost: dict = field(default_factory=dict)
    notes: str = ""
    ok: bool = True
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class Researcher:
    def __init__(self, view: PartyView, own_utility: Utility, program_id: str,
                 registry: Optional[FoxRegistry] = None, runs_dir: Path = Path("runs"),
                 observations: Optional[list[Observation]] = None, seed: int = 0) -> None:
        self.view = view
        self.own_utility = own_utility
        self.program_id = program_id
        self.registry = registry or FoxRegistry()
        self.runs_dir = runs_dir
        self.observations = observations or []
        self.seed = seed

    # ------------------------------------------------------------------ one task
    def run_task(self, task: dict) -> ResearchOutput:
        task_id = str(task.get("task_id", "t??"))
        ident = str(task.get("fox_or_tool", ""))
        started = time.perf_counter()

        if ident in self.registry.catalog:
            return self._run_fox(task_id, ident, started)
        return ResearchOutput(
            task_id=task_id, fox_id=None, version=None, ok=False,
            error=f"`{ident}` is not a catalog fox; tools are called by the planner directly, "
                  f"not by a researcher",
            cost={"seconds": time.perf_counter() - started, "tokens": 0})

    def _run_fox(self, task_id: str, fox_id: str, started: float) -> ResearchOutput:
        entry = self.registry.entry(fox_id)
        status = self.registry.status(fox_id)
        if status == "candidate":
            # Explicit failure, as §8 demands: nothing is substituted or approximated.
            return ResearchOutput(
                task_id=task_id, fox_id=fox_id, version=entry.get("version"), ok=False,
                error=f"`{fox_id}` is declared `candidate`: it is not implemented and is not "
                      f"used. Reason: {entry.get('blocked_by', 'no implementation')}",
                evidence=[f"catalog:{fox_id}"],
                cost={"seconds": time.perf_counter() - started, "tokens": 0})

        try:
            kwargs: dict[str, Any] = {}
            if fox_id.startswith("f06"):
                kwargs["params"] = ["rv"]
            fox = self.registry.create(fox_id, allow_experimental=(status != "validated"),
                                       **kwargs)
            state: BeliefState = fox.init_state(self.view.domain, self.own_utility)
            state.data["counterpart_party"] = self.view.counterpart_role_id
            state.program_id, state.party = self.program_id, self.view.role_id
            for obs in self.observations:
                state = fox.update(state, obs)
            posterior = self.registry.call(fox, state, step="preparation",
                                           experimental=(status != "validated"))
            diagnostics = fox.diagnostics(state)
        except Exception as exc:
            return ResearchOutput(
                task_id=task_id, fox_id=fox_id, version=entry.get("version"), ok=False,
                error=f"{type(exc).__name__}: {exc}",
                cost={"seconds": time.perf_counter() - started, "tokens": 0})

        ref = self._persist(fox_id, posterior)
        notes = []
        if not posterior.scope_ok:
            notes.append("out of scope: its output must NOT enter the pool")
        if not self.observations:
            notes.append("no table observations: what is returned is the prior")
        return ResearchOutput(
            task_id=task_id, fox_id=fox_id, version=posterior.version,
            result={"posterior_ref": str(ref), "summary": posterior.summary()},
            diagnostics={"entropy": diagnostics.entropy, "scope_ok": diagnostics.scope_ok,
                         "n_observations": diagnostics.n_observations,
                         "warnings": diagnostics.warnings + posterior.warnings},
            evidence=([f"paper:{pid}" for pid in (entry.get("paper_ids") or [])]
                      + [f"corpus:{d.doc_id}" for d in self.view.own_docs]
                      + [f"skill:{entry.get('skill_path')}"]),
            cost={"seconds": time.perf_counter() - started, "tokens": 0},
            notes="; ".join(notes) or "no incidents",
            ok=True)

    def _persist(self, fox_id: str, posterior) -> Path:
        target = (self.runs_dir / self.program_id / self.view.role_id / "prep" /
                  f"post_{fox_id}.parquet")
        target.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(posterior.samples, columns=posterior.params)
        frame["weight"] = posterior.weights
        frame.to_parquet(target, index=False)
        return target


def run_plan(plan: list[dict], researcher: Researcher, budget_seconds: float = 120.0,
             max_workers: int = 4) -> list[ResearchOutput]:
    """Launch one researcher per task, in parallel and with a budget."""
    fox_tasks = [t for t in plan if t.get("fox_or_tool") in researcher.registry.catalog]
    outputs: list[ResearchOutput] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(researcher.run_task, task): task for task in fox_tasks}
        for future, task in futures.items():
            try:
                outputs.append(future.result(timeout=budget_seconds))
            except FuturesTimeout:
                outputs.append(ResearchOutput(
                    task_id=str(task.get("task_id", "t??")),
                    fox_id=task.get("fox_or_tool"), version=None, ok=False,
                    error=f"budget exhausted ({budget_seconds:.0f} s)",
                    cost={"seconds": budget_seconds, "tokens": 0}))
    return sorted(outputs, key=lambda o: o.task_id)


def integrate(outputs: list[ResearchOutput], param: str = "rv",
              seed: int = 0) -> dict:
    """The planner's integration: an equal-weight pool of what stayed in scope.

    The planner **cannot edit** a fox's output (protocol §6.5, rule 7): it can only discard it,
    on record. What is discarded here is exactly what the fox itself declared out of scope, and
    it is recorded.
    """
    from foxes.base import Posterior
    from foxes.pooling import disagreement, linear_pool

    usable: list[Posterior] = []
    discarded: list[dict] = []
    for out in outputs:
        if not out.ok:
            discarded.append({"fox_id": out.fox_id, "why": out.error})
            continue
        ref = Path(out.result["posterior_ref"])
        if not ref.exists():
            discarded.append({"fox_id": out.fox_id, "why": "posterior not persisted"})
            continue
        frame = pd.read_parquet(ref)
        if param not in frame.columns:
            continue
        if not out.diagnostics.get("scope_ok", False):
            discarded.append({"fox_id": out.fox_id,
                              "why": "the fox declared itself out of scope"})
            continue
        usable.append(Posterior(out.fox_id or "?", out.version or "?", [param],
                                frame[[param]].to_numpy(), frame["weight"].to_numpy()))
    if not usable:
        return {"param": param, "pooled": None, "discarded": discarded,
                "note": "no fox in scope for this parameter"}
    pooled = linear_pool(usable, param, seed=seed)
    return {"param": param, "pooled_summary": pooled.summary(),
            "n_foxes": len(usable), "foxes": [p.fox_id for p in usable],
            "disagreement_sd": disagreement(usable, param) if len(usable) > 1 else 0.0,
            "discarded": discarded}
