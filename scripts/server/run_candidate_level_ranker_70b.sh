#!/usr/bin/env bash

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
WM_PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
CLEAN_STATS="$ROOT/artifacts/clean_multiseed_llama31_70b_solvability.json"
CONTINUOUS="$ROOT/artifacts/full_dreamer_v3_llama31_70b"
BINARY="$ROOT/artifacts/next_round_70b_ablation/full_binary"
NORANK="$ROOT/artifacts/full_dreamer_v3_llama31_70b_hybrid_lite_norank"
SKLEARN="$ROOT/artifacts/next_round_70b_ablation/sklearn"
ARCHIVE=/share/guozhix/wmagentattack/0710/candidate_level_ranker

cd "$ROOT"
mkdir -p "$ARCHIVE"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

run_ranker() {
  local name="$1"
  local estimator="$2"
  shift 2
  local output="$ARCHIVE/$name"
  mkdir -p "$output"

  PYTHONWARNINGS=ignore "$WM_PY" scripts/26_candidate_level_ranker.py \
    --primary-source norank \
    --seeds 7,13,21 \
    --clean-solvability-json "$CLEAN_STATS" \
    --min-base-success-rate 0.5 \
    --estimator "$estimator" \
    --cv-folds 5 \
    --c-value 0.1 \
    --max-pairs 10000 \
    --output-root "$output" \
    "$@" \
    >"$output/ranker_stdout.json"

  for seed in 7 13 21; do
    for split in val test; do
      PYTHONWARNINGS=ignore "$WM_PY" scripts/18_pareto_utility_selection.py \
        --candidate-json "$output/seed${seed}_${split}_candidates.json" \
        --output "$output/seed${seed}_${split}_pareto.json" \
        --top-k 16,24,32 \
        --seeds 7,13,21 \
        --utility-keys candidate_utility_score,candidate_expected_utility_score,candidate_preservation_score,utility_score,final_utility_score \
        --quantiles 0.50,0.60,0.70,0.80,0.90 \
        --max-per-user-task 2 \
        --clean-solvability-json "$CLEAN_STATS" \
        --min-base-success-rate 0.5 \
        --min-conditional-coverage 0.5 \
        >"$output/seed${seed}_${split}_pareto_stdout.json"
    done
  done

  "$WM_PY" scripts/25_compare_val_selected_transfer.py \
    --report-root "$output" \
    --seeds 7,13,21 \
    --clean-solvability-json "$CLEAN_STATS" \
    --min-base-success-rate 0.5 \
    --min-conditional-coverage 0.5 \
    --output "$output/strict_val_selected_transfer.json" \
    >"$output/strict_stdout.json"
}

# Fixed research budget: one single-source calibration, two Dreamer-only
# multi-view rankers, and one sklearn-assisted upper-bound countercheck.
run_ranker single_pointwise pointwise \
  --source "norank=$NORANK"

run_ranker dreamer_stack_pointwise pointwise \
  --source "norank=$NORANK" \
  --source "continuous=$CONTINUOUS" \
  --source "binary=$BINARY"

run_ranker dreamer_stack_pairwise pairwise \
  --source "norank=$NORANK" \
  --source "continuous=$CONTINUOUS" \
  --source "binary=$BINARY"

run_ranker dreamer_sklearn_upper_bound pointwise \
  --source "norank=$NORANK" \
  --source "continuous=$CONTINUOUS" \
  --source "binary=$BINARY" \
  --source "sklearn=$SKLEARN" \
  --fixed-source-seed sklearn=7

echo "CANDIDATE_LEVEL_RANKER_70B_DONE $(date -Is)"
