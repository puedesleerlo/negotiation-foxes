"""Run the preparation step with the model. By default it does NOT call the model.

  python scripts/run_preparation.py --case parker_gibson                      # dry-run: show only
  python scripts/run_preparation.py --case parker_gibson --role parkers --go   # one party, for real
  python scripts/run_preparation.py --case parker_gibson --go                  # both parties

The dry-run prints the exact prompt and an estimate of input tokens, so the cost is visible
before it is incurred.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.llm import LLMClient  # noqa: E402
from agents.planner import PartyPlanner  # noqa: E402
from gym.case import CaseBundle  # noqa: E402
from gym.prepare import run_preparation  # noqa: E402
from gym.run import make_program_id  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="parker_gibson")
    ap.add_argument("--role", default=None, help="only this party; by default, both")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--go", action="store_true", help="actually call the model")
    ap.add_argument("--show-prompt", action="store_true", help="print the full prompt")
    args = ap.parse_args()

    bundle = CaseBundle.load(args.case)
    roles = [args.role] if args.role else bundle.role_ids()
    program_id = make_program_id(args.case, args.seed, "llm-prep", "")

    if not args.go:
        client = LLMClient()
        print(f"DRY-RUN · case {args.case} · program {program_id}")
        print(f"configured model: {client.model} at {client.endpoint or '(no endpoint)'}\n")
        total = 0
        for role in roles:
            planner = PartyPlanner(bundle.party_view(role), client=client)
            prompt = planner.user_prompt()
            approx = len(prompt) // 3          # ~3 characters per token, rough
            total += approx
            print(f"  {role}: {len(prompt):,} prompt characters (~{approx:,} input tokens)")
            if args.show_prompt:
                print("\n" + "-" * 70 + f"\n{prompt}\n" + "-" * 70 + "\n")
        print(f"\nEstimated input total: ~{total:,} tokens over {len(roles)} call(s).")
        print("Add --go to run it for real.")
        return

    results = run_preparation(args.case, program_id, roles)
    for role, res in results.items():
        status = "OK" if res["ok"] else "FAILED VALIDATION"
        print(f"\n=== {role}: {status}")
        usage = res["usage"]
        print(f"  tokens: {usage.get('prompt_tokens')} input + "
              f"{usage.get('completion_tokens')} output "
              f"({usage.get('reasoning_tokens')} reasoning) · "
              f"{usage.get('seconds', 0):.1f} s")
        if not res["ok"]:
            for err in res["validation"]["errors"]:
                print(f"  - {err}")
            continue
        val = res["validation"]
        print(f"  plan: {val['n_plan_tasks']} tasks · hypotheses: {val['n_hypotheses']} · "
              f"foxes rejected on scope: {val['n_rejected_foxes']}")
        for warning in val.get("warnings", []):
            print(f"  warning: {warning}")
        print(f"  artifacts in runs/{program_id}/{role}/prep/")


if __name__ == "__main__":
    main()
