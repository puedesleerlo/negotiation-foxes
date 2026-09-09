"""Common fox CLI: run one fox over a JSONL trace of observations.

Usage:
  python -m foxes.cli --fox f01_bayes_rv_concession --domain d2_case3 --obs trace.jsonl
  python -m foxes.cli --list

Each JSONL line is an observation: {"round":3,"party":"cp","offer":{...},
"est_utility_to_proposer":0.82,"max_rounds":20}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from foxes.base import Observation
from foxes.registry import FoxRegistry
from foxes.synthetic import DOMAINS


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a fox over a trace of observations")
    ap.add_argument("--fox")
    ap.add_argument("--domain", default="d2_case3", choices=sorted(DOMAINS))
    ap.add_argument("--obs", help="JSONL of observations")
    ap.add_argument("--experimental", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    reg = FoxRegistry()
    if args.list or not args.fox:
        for fox_id, entry in sorted(reg.catalog.items()):
            print(f"{fox_id:34} {entry['status']:12} estimates {entry.get('estimates')}")
        return

    fox = reg.create(args.fox, allow_experimental=args.experimental)
    state = fox.init_state(DOMAINS[args.domain])
    state.data["counterpart_party"] = "cp"
    if args.obs:
        for line in Path(args.obs).read_text().splitlines():
            if line.strip():
                state = fox.update(state, Observation(**json.loads(line)))
    post = reg.call(fox, state, step="cli", experimental=args.experimental)
    print(json.dumps(post.summary(), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
