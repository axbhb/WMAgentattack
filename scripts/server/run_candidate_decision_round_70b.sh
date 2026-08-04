#!/usr/bin/env bash

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
WM_PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
CLEAN_STATS="$ROOT/artifacts/clean_multiseed_llama31_70b_solvability.json"
CONTINUOUS="$ROOT/artifacts/full_dreamer_v3_llama31_70b"
BINARY="$ROOT/artifacts/next_round_70b_ablation/full_binary"
NORANK="$ROOT/artifacts/full_dreamer_v3_llama31_70b_hybrid_lite_norank"
FIRST_ROUND=/share/guozhix/wmagentattack/0710/candidate_level_ranker
ARCHIVE=/share/guozhix/wmagentattack/0710/candidate_decision_round
UTILITY_KEYS=candidate_utility_score,candidate_expected_utility_score,candidate_preservation_score,utility_score,final_utility_score

cd "$ROOT"
mkdir -p "$ARCHIVE"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

run_selection() {
  local output="$1"
  local seeds="$2"
  local seed_list="$3"
  for seed in $seed_list; do
    for split in val test; do
      PYTHONWARNINGS=ignore "$WM_PY" scripts/18_pareto_utility_selection.py \
        --candidate-json "$output/seed${seed}_${split}_candidates.json" \
        --output "$output/seed${seed}_${split}_pareto.json" \
        --top-k 16,24,32 \
        --seeds 7,13,21 \
        --utility-keys "$UTILITY_KEYS" \
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
    --seeds "$seeds" \
    --clean-solvability-json "$CLEAN_STATS" \
    --min-base-success-rate 0.5 \
    --min-conditional-coverage 0.5 \
    --output "$output/strict_val_selected_transfer.json" \
    >"$output/strict_stdout.json"
}

run_consensus() {
  local input="$1"
  local output="$2"
  mkdir -p "$output"
  "$WM_PY" scripts/27_seed_ensemble_candidates.py \
    --input-root "$input" \
    --seeds 7,13,21 \
    --output-seed 7 \
    --output-root "$output" \
    >"$output/ensemble_stdout.json"
  run_selection "$output" 7 "7"
}

run_ranker() {
  local name="$1"
  local joint_estimator="$2"
  local output="$ARCHIVE/$name"
  mkdir -p "$output"
  PYTHONWARNINGS=ignore "$WM_PY" scripts/26_candidate_level_ranker.py \
    --primary-source norank \
    --source "norank=$NORANK" \
    --source "continuous=$CONTINUOUS" \
    --source "binary=$BINARY" \
    --seeds 7,13,21 \
    --clean-solvability-json "$CLEAN_STATS" \
    --min-base-success-rate 0.5 \
    --estimator pointwise \
    --joint-estimator "$joint_estimator" \
    --test-prediction-mode crossfit_ensemble \
    --cv-folds 5 \
    --c-value 0.1 \
    --max-pairs 10000 \
    --output-root "$output" \
    >"$output/ranker_stdout.json"
  run_selection "$output" 7,13,21 "7 13 21"
  run_consensus "$output" "$output/seed_ensemble"
}

# Formalize the already-completed full-fit pointwise consensus before running
# the second fixed budget.
run_consensus \
  "$FIRST_ROUND/dreamer_stack_pointwise" \
  "$ARCHIVE/fullfit_stack_seed_ensemble"

# Second fixed budget: aligned cross-fit prediction scales, then two direct
# ASR+BUP objectives. No hyperparameter search is performed.
run_ranker crossfit_marginal marginal_sum
run_ranker crossfit_joint_pairwise ordinal_pairwise
run_ranker crossfit_joint_ridge ridge

echo "CANDIDATE_DECISION_ROUND_70B_DONE $(date -Is)"
