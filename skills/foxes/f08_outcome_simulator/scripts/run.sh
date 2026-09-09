#!/usr/bin/env bash
# Run this fox on a JSONL trace of observations.
# Usage: bash skills/foxes/f08_outcome_simulator/scripts/run.sh <trace.jsonl> [domain]
set -euo pipefail
cd "$(dirname "$0")/../../../.."
PYTHONPATH=. .venv/bin/python -m foxes.cli --fox f08_outcome_simulator --obs "$1" --domain "${2:-d2_case3}" --experimental
