"""Orchestration of the preparation step (task 2.4).

Instantiates one planner per role with its own context, in isolation, produces the artifacts of
§5.1 and seals the declared utility. The two planners do not communicate.

Artifacts per party, in `runs/<program_id>/<role_id>/prep/`:
  strategy.json / strategy.md   the strategy with its three blocks
  plan.json                     tasks with the exact fox or tool and its scope justification
  declared_utility.json         what the gym seals
  hypotheses.jsonl              hypotheses registered on the platform
  preparation_memo.md           readable memo with quotes
  research.json                 the researchers' typed outputs
  integration.json              the pooled belief and what was discarded, with reasons
  distributions.parquet         one row per sample per fox, with its weight
  outcomes.json                 ZOPA, expected utility and EVPI under the integrated belief
  llm_calls.jsonl               raw response of every call, so the step can be redone
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from agents.declared_utility import DeclaredUtility, derive_from_view
from agents.llm import LLMClient
from agents.planner import PartyPlanner, PlannerResult
from agents.researcher import Researcher, integrate, run_plan
from gym.case import CaseBundle, PartyView
from gym.isolation import LeakDetector
from gym.trace import Trace
from hypotheses.schema import HypothesisTest, Threshold
from hypotheses.store import HypothesisStore
from tools.decision import evpi

VALID_TEST_KINDS = {"probe_offer", "observe_rounds", "meso", "contingent_clause"}


def strategy_markdown(role_id: str, strategy: dict, validation: dict) -> str:
    """`strategy.md` with the three blocks §5.1 asks for."""
    view = strategy.get("case_view", {})
    params = (strategy.get("strategy", {}) or {}).get("negotiation_parameters", {}) or {}
    lines = [f"# Strategy — {role_id}", "",
             "Produced by the planner in the preparation step and **frozen** by the gym before "
             "the first round. Deviations during the negotiation are justified in each turn's "
             "memo and counted by the feedback.", "",
             "## 1. Features", ""]
    for item in (strategy.get("strategy", {}) or {}).get("features", []) or []:
        lines.append(f"- {item}")
    lines += ["", "### Own interests", ""]
    for item in view.get("my_interests", []) or []:
        lines.append(f"- {item}")
    lines += ["", "### Inferred counterpart interests", ""]
    for item in view.get("inferred_counterpart_interests", []) or []:
        if isinstance(item, dict):
            lines.append(f"- {item.get('claim')} — *basis*: {item.get('basis')} "
                         f"(confidence {item.get('confidence')})")
        else:
            lines.append(f"- {item}")
    lines += ["", "### Sources of uncertainty", ""]
    for item in view.get("uncertainty_sources", []) or []:
        lines.append(f"- {item}")

    declared = strategy.get("declared_utility", {}) or {}
    rv = declared.get("reservation_value", {}) or {}
    lines += ["", "## 2. Negotiation parameters", "",
              f"- **Reservation value**: {rv.get('value')} — quote: «{rv.get('quote', '')}»",
              f"- **Aspiration**: {(declared.get('aspiration') or {}).get('value')} — "
              f"{(declared.get('aspiration') or {}).get('reasoning', '')}",
              f"- **BATNA**: {(declared.get('batna') or {}).get('description', '')}",
              f"- **First offer**: {(params.get('first_offer') or {}).get('value')} — "
              f"{(params.get('first_offer') or {}).get('reasoning', '')}",
              f"- **Concession plan**: {params.get('concession_plan', '')}",
              f"- **Walk-away rule**: {params.get('walk_away_rule', '')}",
              "", "## 3. Approach and strategies", ""]
    for step in (strategy.get("strategy", {}) or {}).get("approach", []) or []:
        lines.append(f"1. {step}")
    lines += ["", "### Uncertainty-exploitation levers", ""]
    for lever in (strategy.get("strategy", {}) or {}).get("uncertainty_levers", []) or []:
        lines.append(f"- **{lever.get('lever')}**: {lever.get('use')} — {lever.get('why')}")

    lines += ["", "## 4. Foxes and tools I will use", "",
              "| task | fox or tool | phase | why it is in scope |", "|---|---|---|---|"]
    for task in strategy.get("plan", []) or []:
        lines.append(f"| {task.get('task_id')} | `{task.get('fox_or_tool')}` | "
                     f"{task.get('phase')} | {task.get('why_in_scope')} |")
    rejected = strategy.get("foxes_rejected", []) or []
    if rejected:
        lines += ["", "### Rejected on scope", ""]
        for item in rejected:
            lines.append(f"- `{item.get('id')}`: {item.get('why')}")
    if validation.get("warnings"):
        lines += ["", "### Validation warnings", ""]
        for warning in validation["warnings"]:
            lines.append(f"- {warning}")
    return "\n".join(lines) + "\n"


def preparation_memo(role_id: str, strategy: dict, declared: DeclaredUtility,
                     result: PlannerResult) -> str:
    rv = (strategy.get("declared_utility", {}) or {}).get("reservation_value", {}) or {}
    agree = (abs(float(rv.get("value", -1)) - declared.reservation_value_raw) < 1e-6
             if isinstance(rv.get("value"), (int, float)) else False)
    lines = [f"# Preparation memo — {role_id}", "",
             "## What I know about my own position", "",
             f"My reservation value is **{rv.get('value')}**, and I know it from this sentence "
             f"of my confidential material:", "", f"> {rv.get('quote', '')}", "",
             f"The gym's deterministic extraction (`agents/declared_utility.py`) reaches the "
             f"same number independently: **{declared.reservation_value_raw:.0f}**. "
             + ("The two readings agree, so the value is well anchored."
                if agree else
                "**They do not agree**: one of the two readings must be checked before "
                "sealing anything."),
             "", "## What I do not know about the counterpart", ""]
    for item in (strategy.get("case_view", {}) or {}).get("uncertainty_sources", []) or []:
        lines.append(f"- {item}")
    lines += ["", "None of those unknowns is estimated by me: the plan's foxes estimate them, "
              "and they are calibrated models with a declared scope. What I decide is which "
              "ones apply.", "", "## What I will do and with what", ""]
    for task in strategy.get("plan", []) or []:
        lines.append(f"- **{task.get('task_id')}** — {task.get('objective')} "
                     f"with `{task.get('fox_or_tool')}` ({task.get('phase')}). "
                     f"In scope because: {task.get('why_in_scope')}")
    lines += ["", "## Hypotheses I register", ""]
    for hyp in strategy.get("hypotheses", []) or []:
        lines.append(f"- {hyp.get('statement')} — resolves like this: "
                     f"{(hyp.get('test') or {}).get('resolves_if')}")
    lines += ["", "## Cost of this step", "",
              f"- {result.usage.get('calls', 1)} model call(s) · "
              f"{result.usage.get('prompt_tokens')} input tokens + "
              f"{result.usage.get('completion_tokens')} output "
              f"({result.usage.get('reasoning_tokens')} reasoning) · "
              f"{result.usage.get('seconds', 0):.1f} s", ""]
    return "\n".join(lines)


def register_hypotheses(store: HypothesisStore, role_id: str, strategy: dict) -> list[str]:
    ids: list[str] = []
    for hyp in strategy.get("hypotheses", []) or []:
        test_raw = hyp.get("test") or {}
        kind = test_raw.get("kind")
        if kind not in VALID_TEST_KINDS:
            kind = "observe_rounds"
        threshold = None
        raw_threshold = hyp.get("threshold") or {}
        if raw_threshold.get("comparison") and raw_threshold.get("values"):
            try:
                threshold = Threshold(comparison=raw_threshold["comparison"],
                                      values=[float(v) for v in raw_threshold["values"]],
                                      unit=str(raw_threshold.get("unit", "USD")))
            except Exception:
                threshold = None
        item = store.register(
            party=role_id, type="parameter" if str(hyp.get("variable", "")).startswith("theta")
            else "behavior",
            statement=str(hyp.get("statement", ""))[:400],
            variable=hyp.get("variable"),
            prior_belief=float(hyp.get("prior_belief", 0.5)),
            evidence=["planner"],
            test=HypothesisTest(kind=kind, spec=test_raw.get("spec") or {},
                                resolves_if=str(test_raw.get("resolves_if", ""))[:400]),
            created_at="prep")
        if threshold is not None:
            item.threshold = threshold
        ids.append(item.hypothesis_id)
    store.save()
    return ids


def prepare_party(view: PartyView, program_id: str, trace: Trace, client: LLMClient,
                  runs_dir: Path) -> dict:
    """Preparation of *one* party. It receives nothing of the other."""
    declared = derive_from_view(view)
    planner = PartyPlanner(view, client=client)
    result = planner.plan()

    trace.write_artifact(f"{view.role_id}/prep/llm_calls.jsonl",
                         json.dumps({"step": "planner", "usage": result.usage,
                                     "text": result.raw_text, "reasoning": result.reasoning},
                                    ensure_ascii=False) + "\n", party=view.role_id)
    if not result.ok:
        trace.emit("preparation_failed", party=view.role_id, step="preparation",
                   errors=result.validation.get("errors"))
        return {"role_id": view.role_id, "ok": False, "validation": result.validation,
                "usage": result.usage}

    store = HypothesisStore(program_id, view.role_id, runs_dir=runs_dir)
    hypothesis_ids = register_hypotheses(store, view.role_id, result.strategy)

    # --- §5.1 step 4: one researcher per plan task, in parallel and with a budget.
    researcher = Researcher(view, declared.to_utility(view), program_id, runs_dir=runs_dir)
    research = run_plan(result.strategy.get("plan", []), researcher)
    trace.write_json(f"{view.role_id}/prep/research.json",
                     [out.to_dict() for out in research], party=view.role_id)

    # --- step 5: integration. The planner cannot edit a fox's output; only the pool of what
    # stayed in scope, with a record of what was discarded (rule 7 of §6.5).
    pooled = integrate(research, param="rv")
    distributions = _write_distributions(view, program_id, research, runs_dir)
    outcomes = _outcomes(view, declared, pooled, runs_dir, program_id)
    trace.write_json(f"{view.role_id}/prep/outcomes.json", outcomes, party=view.role_id)
    trace.write_json(f"{view.role_id}/prep/integration.json", pooled, party=view.role_id)

    trace.write_json(f"{view.role_id}/prep/strategy.json", result.strategy, party=view.role_id)
    trace.write_artifact(f"{view.role_id}/prep/strategy.md",
                         strategy_markdown(view.role_id, result.strategy, result.validation),
                         party=view.role_id)
    trace.write_json(f"{view.role_id}/prep/plan.json",
                     {"plan": result.strategy.get("plan", []),
                      "foxes_rejected": result.strategy.get("foxes_rejected", [])},
                     party=view.role_id)
    trace.write_json(f"{view.role_id}/prep/declared_utility.json", declared.to_dict(),
                     party=view.role_id)
    trace.write_artifact(f"{view.role_id}/prep/preparation_memo.md",
                         preparation_memo(view.role_id, result.strategy, declared, result),
                         party=view.role_id)
    trace.emit("preparation", party=view.role_id, step="preparation",
               declared_utility=declared.to_dict(), validation=result.validation,
               hypotheses=hypothesis_ids, usage=result.usage,
               research=[{"task_id": o.task_id, "fox_id": o.fox_id, "ok": o.ok,
                          "scope_ok": o.diagnostics.get("scope_ok"), "error": o.error}
                         for o in research],
               integration={k: v for k, v in pooled.items() if k != "pooled_summary"},
               outcomes=outcomes)
    return {"role_id": view.role_id, "ok": True, "declared": declared,
            "strategy": result.strategy, "validation": result.validation,
            "usage": result.usage, "hypotheses": hypothesis_ids,
            "research": research, "integration": pooled, "outcomes": outcomes,
            "distributions": distributions}


def _write_distributions(view: PartyView, program_id: str, research, runs_dir: Path) -> str:
    """`distributions.parquet`: one row per sample and per fox, with its weight."""
    frames = []
    for out in research:
        ref = out.result.get("posterior_ref") if out.ok else None
        if not ref or not Path(ref).exists():
            continue
        frame = pd.read_parquet(ref)
        frame["fox_id"] = out.fox_id
        frame["scope_ok"] = bool(out.diagnostics.get("scope_ok"))
        frames.append(frame)
    target = runs_dir / program_id / view.role_id / "prep" / "distributions.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    if frames:
        pd.concat(frames, ignore_index=True).to_parquet(target, index=False)
    return str(target)


def _outcomes(view: PartyView, declared: DeclaredUtility, pooled: dict, runs_dir: Path,
              program_id: str) -> dict:
    """`outcomes.json`: ZOPA, expected utility and EVPI per round under the integrated belief."""
    utility = declared.to_utility(view)
    space = view.domain.outcome_space()
    rng = np.random.default_rng(0)
    n = 2000
    summary = pooled.get("pooled_summary")
    if summary:
        ref = runs_dir / program_id / view.role_id / "prep" / "distributions.parquet"
        frame = pd.read_parquet(ref)
        usable = frame[frame.scope_ok & frame["rv"].notna()] if "rv" in frame.columns else frame
        rv = (usable["rv"].sample(n, replace=True, random_state=0).to_numpy()
              if len(usable) else rng.uniform(0, 1, n))
    else:
        rv = rng.uniform(0, 1, n)
    theta = {"rv": rv, "beta": rng.uniform(0.2, 3.0, n), "T": rng.uniform(10, 30, n)}
    horizon = int(view.protocol.get("max_rounds", 20))
    by_round = {str(r): float(evpi(space, utility, r, theta)["evpi"])
                for r in (1, max(horizon // 2, 1), horizon)}
    first = evpi(space, utility, 1, theta)
    return {
        "belief_source": pooled.get("foxes", []),
        "discarded": pooled.get("discarded", []),
        "rv_belief": {"median": float(np.median(rv)),
                      "q05": float(np.quantile(rv, 0.05)),
                      "q95": float(np.quantile(rv, 0.95))},
        "evpi_rv_by_round": by_round,
        "best_offer_ex_ante": first["best_offer_ex_ante"],
        "expected_utility_ex_ante": first["expected_utility_ex_ante"],
    }


def run_preparation(case_id: str, program_id: str, roles: Optional[list[str]] = None,
                    runs_dir: Path = Path("runs"),
                    client: Optional[LLMClient] = None) -> dict:
    bundle = CaseBundle.load(case_id)
    detector = LeakDetector.for_case(bundle)
    trace = Trace(program_id, root=runs_dir, detector=detector)
    client = client or LLMClient()
    out: dict = {}
    for role_id in (roles or bundle.role_ids()):
        out[role_id] = prepare_party(bundle.party_view(role_id), program_id, trace, client,
                                     runs_dir)
    trace.emit("sealed", step="setup",
               note="declared utilities frozen at the close of preparation",
               isolation=detector.summary())
    return out
