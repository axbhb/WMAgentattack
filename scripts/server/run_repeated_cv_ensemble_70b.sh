#!/usr/bin/env bash

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
WM_PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
CLEAN_STATS="$ROOT/artifacts/clean_multiseed_llama31_70b_solvability.json"
ROBUST=/share/guozhix/wmagentattack/0710/candidate_ranker_robustness
OUTPUT="$ROBUST/repeated_cv_seed_ensemble"
UTILITY_KEYS=candidate_utility_score,candidate_expected_utility_score,candidate_preservation_score,utility_score,final_utility_score

cd "$ROOT"
mkdir -p "$OUTPUT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

"$WM_PY" scripts/28_repeated_cv_ensemble.py \
  --input-root "$ROBUST/cv_20260710/seed_ensemble" \
  --input-root "$ROBUST/cv_20260711/seed_ensemble" \
  --input-root "$ROBUST/cv_20260712/seed_ensemble" \
  --input-root "$ROBUST/cv_20260713/seed_ensemble" \
  --input-root "$ROBUST/cv_20260714/seed_ensemble" \
  --input-seed 7 \
  --output-seed 7 \
  --output-root "$OUTPUT" \
  >"$OUTPUT/ensemble_stdout.json"

for split in val test; do
  "$WM_PY" scripts/18_pareto_utility_selection.py \
    --candidate-json "$OUTPUT/seed7_${split}_candidates.json" \
    --output "$OUTPUT/seed7_${split}_pareto.json" \
    --top-k 16,24,32 \
    --seeds 7,13,21 \
    --utility-keys "$UTILITY_KEYS" \
    --quantiles 0.50,0.60,0.70,0.80,0.90 \
    --max-per-user-task 2 \
    --clean-solvability-json "$CLEAN_STATS" \
    --min-base-success-rate 0.5 \
    --min-conditional-coverage 0.5 \
    >"$OUTPUT/seed7_${split}_pareto_stdout.json"
done

"$WM_PY" scripts/25_compare_val_selected_transfer.py \
  --report-root "$OUTPUT" \
  --seeds 7 \
  --clean-solvability-json "$CLEAN_STATS" \
  --min-base-success-rate 0.5 \
  --min-conditional-coverage 0.5 \
  --utility-key-priority final_utility_score,utility_score,candidate_preservation_score,candidate_utility_score,candidate_expected_utility_score \
  --output "$OUTPUT/strict_val_selected_transfer.json" \
  >"$OUTPUT/strict_stdout.json"

echo "REPEATED_CV_ENSEMBLE_70B_DONE $(date -Is)"
