#!/usr/bin/env bash
# Run this fox on a JSONL trace of observations.
# Usage: bash skills/foxes/f10_impasse_risk/scripts/run.sh <trace.jsonl> [domain]
set -euo pipefail
cd "$(dirname "$0")/../../../.."
PYTHONPATH=. .venv/bin/python -m foxes.cli --fox f10_impasse_risk --obs "$1" --domain "${2:-d2_case3}" --experimental
