#!/usr/bin/env bash
#SBATCH --job-name=wma-semres-event
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%j.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
DATA=/share/guozhix/wmagentattack/0721/factorized_refactor/fixed_budget_v1/data/v2_joint_labels
PREVIOUS=/share/guozhix/wmagentattack/0721/factorized_refactor/fixed_budget_v1/summary.json
ARCHIVE=/share/guozhix/wmagentattack/0722/semantic_residual_event/fixed_budget_v1

cd "$ROOT"
test ! -e "$ARCHIVE/COMPLETE"
mkdir -p "$ARCHIVE/model" logs
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

failure_record() {
  status=$?
  {
    echo "failed_at=$(date --iso-8601=seconds)"
    echo "exit_status=$status"
  } > "$ARCHIVE/FAILED"
  exit "$status"
}
trap failure_record ERR

{
  echo "job_id=$SLURM_JOB_ID"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "git_head=$(git rev-parse HEAD)"
  "$PY" --version
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
} > "$ARCHIVE/runtime.txt"

cp configs/0722_semantic_residual_event_protocol.json \
  "$ARCHIVE/protocol.preregistered.json"
sha256sum \
  configs/0722_semantic_residual_event_protocol.json \
  src/wmagentattack/semantic_residual_event_model.py \
  scripts/113_train_semantic_residual_event_model.py \
  scripts/114_summarize_semantic_residual_round.py \
  tests/test_semantic_residual_event_model.py \
  tests/test_semantic_residual_event_summary.py \
  > "$ARCHIVE/code.prerun.sha256"
sha256sum "$DATA"/*.jsonl "$DATA"/audit.json > "$ARCHIVE/data.prerun.sha256"

"$PY" -m pytest -q \
  tests/test_semantic_residual_event_model.py \
  tests/test_semantic_residual_event_summary.py \
  > "$ARCHIVE/preflight_tests.txt"

for SEED in 7 17 29; do
  "$PY" scripts/113_train_semantic_residual_event_model.py \
    --train "$DATA/train_trajectories.jsonl" \
    --validation "$DATA/val_trajectories.jsonl" \
    --test "$DATA/test_trajectories.jsonl" \
    --train-metadata "$DATA/train_metadata.jsonl" \
    --validation-metadata "$DATA/val_metadata.jsonl" \
    --test-metadata "$DATA/test_metadata.jsonl" \
    --output-dir "$ARCHIVE/model/seed$SEED" \
    --epochs 12 \
    --batch-size 64 \
    --learning-rate 0.0003 \
    --hidden-size 64 \
    --layers 2 \
    --heads 4 \
    --seed "$SEED" \
    --max-generation-steps 16 \
    --minimum-support-probability 0.35 \
    --device cuda \
    > "$ARCHIVE/model/seed${SEED}_stdout.json"
done

"$PY" scripts/114_summarize_semantic_residual_round.py \
  --root "$ARCHIVE" \
  --previous-summary "$PREVIOUS" \
  --output "$ARCHIVE/summary.json" \
  > "$ARCHIVE/summary_stdout.json"

{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "runtime_failures=0"
} >> "$ARCHIVE/runtime.txt"
find "$ARCHIVE" -type f -not -name checksums.sha256 -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$ARCHIVE/checksums.sha256"
date --iso-8601=seconds > "$ARCHIVE/COMPLETE"
trap - ERR
