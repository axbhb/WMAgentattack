#!/usr/bin/env bash

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
WM_PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
CLEAN_STATS="$ROOT/artifacts/clean_multiseed_llama31_70b_solvability.json"
CONTINUOUS="$ROOT/artifacts/full_dreamer_v3_llama31_70b"
BINARY="$ROOT/artifacts/next_round_70b_ablation/full_binary"
NORANK="$ROOT/artifacts/full_dreamer_v3_llama31_70b_hybrid_lite_norank"
ARCHIVE=/share/guozhix/wmagentattack/0710/candidate_ranker_robustness
UTILITY_KEYS=candidate_utility_score,candidate_expected_utility_score,candidate_preservation_score,utility_score,final_utility_score
CV_STATES=(20260710 20260711 20260712 20260713 20260714)

cd "$ROOT"
mkdir -p "$ARCHIVE"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

select_consensus() {
  local output="$1"
  for split in val test; do
    "$WM_PY" scripts/18_pareto_utility_selection.py \
      --candidate-json "$output/seed7_${split}_candidates.json" \
      --output "$output/seed7_${split}_pareto.json" \
      --top-k 16,24,32 \
      --seeds 7,13,21 \
      --utility-keys "$UTILITY_KEYS" \
      --quantiles 0.50,0.60,0.70,0.80,0.90 \
      --max-per-user-task 2 \
      --clean-solvability-json "$CLEAN_STATS" \
      --min-base-success-rate 0.5 \
      --min-conditional-coverage 0.5 \
      >"$output/seed7_${split}_pareto_stdout.json"
  done
  "$WM_PY" scripts/25_compare_val_selected_transfer.py \
    --report-root "$output" \
    --seeds 7 \
    --clean-solvability-json "$CLEAN_STATS" \
    --min-base-success-rate 0.5 \
    --min-conditional-coverage 0.5 \
    --output "$output/strict_val_selected_transfer.json" \
    >"$output/strict_stdout.json"
}

for state in "${CV_STATES[@]}"; do
  run="$ARCHIVE/cv_${state}"
  consensus="$run/seed_ensemble"
  mkdir -p "$run" "$consensus"
  PYTHONWARNINGS=ignore "$WM_PY" scripts/26_candidate_level_ranker.py \
    --primary-source norank \
    --source "norank=$NORANK" \
    --source "continuous=$CONTINUOUS" \
    --source "binary=$BINARY" \
    --seeds 7,13,21 \
    --clean-solvability-json "$CLEAN_STATS" \
    --min-base-success-rate 0.5 \
    --estimator pointwise \
    --joint-estimator marginal_sum \
    --test-prediction-mode crossfit_ensemble \
    --cv-folds 5 \
    --c-value 0.1 \
    --max-pairs 10000 \
    --random-state "$state" \
    --output-root "$run" \
    >"$run/ranker_stdout.json"
  "$WM_PY" scripts/27_seed_ensemble_candidates.py \
    --input-root "$run" \
    --seeds 7,13,21 \
    --output-seed 7 \
    --output-root "$consensus" \
    >"$consensus/ensemble_stdout.json"
  select_consensus "$consensus"
done

# Leave-one-model-seed-out sensitivity uses the predeclared primary CV state.
PRIMARY="$ARCHIVE/cv_20260710"
for subset in 7,13 7,21 13,21; do
  tag="${subset//,/_}"
  output="$ARCHIVE/model_subset_${tag}"
  mkdir -p "$output"
  "$WM_PY" scripts/27_seed_ensemble_candidates.py \
    --input-root "$PRIMARY" \
    --seeds "$subset" \
    --output-seed 7 \
    --output-root "$output" \
    >"$output/ensemble_stdout.json"
  select_consensus "$output"
done

echo "CANDIDATE_RANKER_ROBUSTNESS_70B_DONE $(date -Is)"
