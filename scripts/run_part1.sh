#!/usr/bin/env bash
# Reproduce Part 1 from scratch: knowledge base, catalog, synthetic dataset, fox calibration and
# validation.
#
#   bash scripts/run_part1.sh            # full (~25 min: downloads papers and generates 6000 episodes)
#   bash scripts/run_part1.sh --fast     # quick (~4 min: 300 episodes per domain)
#
# Requirements: .venv created with `uv venv --python 3.11` and the dependencies installed.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=.venv/bin/python
FAST=${1:-}
# The sample sizes are the ones that produced the numbers cited in catalog/foxes.yaml.
EPISODES=2000; CAL=60; CAL9=40
V_F01=400; V_F02=250; V_F06=250; V_F07=120; V_F04=500; V_F03=150; V_F09=60
if [ "$FAST" = "--fast" ]; then
  EPISODES=300; CAL=30; CAL9=15
  V_F01=80; V_F02=60; V_F06=60; V_F07=40; V_F04=80; V_F03=40; V_F09=20
fi

echo "== 1.1 Kosmos ingestion =="
$PY -m kb.ingest_kosmos
echo "== 1.2 metadata resolution (uses the local cache; network on the first run) =="
$PY -m kb.resolve_metadata
echo "== 1.3 PDF download through legitimate routes =="
$PY -m kb.fetch_papers
echo "== 1.4 text extraction and database =="
$PY -m kb.extract_text
$PY -m kb.build_db
echo "== 1.5 structured annotation (every evidence pointer is validated before writing) =="
$PY -m kb.load_annotations
$PY -m kb.review_queue

echo "== 1.9 synthetic dataset =="
$PY -m foxes.synthetic --episodes "$EPISODES" --seed 20260905

echo "== 1.10a calibration (calibration split only) =="
$PY -m foxes.calibrate --fox f02_concession_issue_weights --episodes 120
$PY -m foxes.calibrate --fox f03_time_concession_regression --episodes "$CAL"
$PY -m foxes.calibrate --fox f04_accept_boundary_kde --episodes 300
$PY -m foxes.calibrate --fox f09_hypothesis_issue_weights --episodes "$CAL9"

echo "== 1.10b validation (validation split only) =="
$PY -m foxes.validate --fox f01_bayes_rv_concession        --episodes "$V_F01"
$PY -m foxes.validate --fox f02_concession_issue_weights  --episodes "$V_F02"
$PY -m foxes.validate --fox f03_time_concession_regression --episodes "$V_F03"
$PY -m foxes.validate --fox f04_accept_boundary_kde       --episodes "$V_F04"
$PY -m foxes.validate --fox f06_uninformed_prior          --episodes "$V_F06"
$PY -m foxes.validate --fox f07_zopa_pareto_estimator     --episodes "$V_F07"
$PY -m foxes.validate --fox f09_hypothesis_issue_weights  --episodes "$V_F09"

echo "== tests =="
$PY -m pytest foxes/tests -q

echo "== report =="
$PY -m kb.report

echo "== belief-chain demo =="
$PY scripts/demo_belief.py --family conceder --domain d2_case3 --seed 3

echo
echo "Done. Knowledge base status: knowledge/db/kb_report.md"
echo "Human review queue:          knowledge/db/review_queue.md"
echo "Validations:                 foxes/<fox_id>/validation_report.md"
