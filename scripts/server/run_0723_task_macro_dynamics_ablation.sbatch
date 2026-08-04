#!/usr/bin/env bash
#SBATCH --job-name=wma-taskmacro-ablate
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
ARCHIVE=/share/guozhix/wmagentattack/0723/task_macro_dynamics_ablation/fixed_budget_v1

cd "$ROOT"
if [[ -e "$ARCHIVE" ]]; then
  echo "Refusing to overwrite frozen archive: $ARCHIVE" >&2
  exit 2
fi
mkdir -p "$ARCHIVE/model" "$ARCHIVE/audits" logs
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

cp configs/0723_task_macro_dynamics_ablation_protocol.json \
  "$ARCHIVE/protocol.preregistered.json"
cp configs/0723_event_ontology_v2_3.json \
  "$ARCHIVE/event_ontology.preregistered.json"
sha256sum \
  configs/0723_task_macro_dynamics_ablation_protocol.json \
  configs/0723_event_ontology_v2_3.json \
  src/wmagentattack/event_ontology.py \
  src/wmagentattack/semantic_residual_event_model.py \
  scripts/116_audit_frozen_event_ontology.py \
  scripts/117_train_task_macro_event_ablation.py \
  scripts/118_summarize_task_macro_dynamics_ablation.py \
  tests/test_event_ontology.py \
  tests/test_task_macro_event_ablation.py \
  tests/test_task_macro_dynamics_summary.py \
  > "$ARCHIVE/code.prerun.sha256"
sha256sum "$DATA"/*.jsonl "$DATA"/audit.json > "$ARCHIVE/data.prerun.sha256"

"$PY" -m pytest -q \
  tests/test_event_ontology.py \
  tests/test_task_macro_event_ablation.py \
  tests/test_task_macro_dynamics_summary.py \
  > "$ARCHIVE/preflight_tests.txt"

"$PY" scripts/116_audit_frozen_event_ontology.py \
  --train "$DATA/train_trajectories.jsonl" \
  --validation "$DATA/val_trajectories.jsonl" \
  --test "$DATA/test_trajectories.jsonl" \
  --output "$ARCHIVE/audits/event_ontology_audit.json" \
  > "$ARCHIVE/audits/event_ontology_audit_stdout.json"

for VARIANT in length_semantic event_no_attack_semantics semantic_event; do
  for SEED in 7 17 29; do
    EXTRA=()
    if [[ "$VARIANT" == "semantic_event" ]]; then
      EXTRA+=(--run-prefix-controls)
    fi
    "$PY" scripts/117_train_task_macro_event_ablation.py \
      --train "$DATA/train_trajectories.jsonl" \
      --validation "$DATA/val_trajectories.jsonl" \
      --test "$DATA/test_trajectories.jsonl" \
      --train-metadata "$DATA/train_metadata.jsonl" \
      --validation-metadata "$DATA/val_metadata.jsonl" \
      --test-metadata "$DATA/test_metadata.jsonl" \
      --output-dir "$ARCHIVE/model/$VARIANT/seed$SEED" \
      --variant "$VARIANT" \
      --epochs 12 \
      --batch-size 64 \
      --learning-rate 0.0003 \
      --hidden-size 64 \
      --layers 2 \
      --heads 4 \
      --seed "$SEED" \
      --max-generation-steps 16 \
      --device cuda \
      "${EXTRA[@]}" \
      > "$ARCHIVE/model/${VARIANT}_seed${SEED}_stdout.json"
  done
done

"$PY" scripts/118_summarize_task_macro_dynamics_ablation.py \
  --root "$ARCHIVE" \
  --protocol configs/0723_task_macro_dynamics_ablation_protocol.json \
  --ontology-audit "$ARCHIVE/audits/event_ontology_audit.json" \
  --output "$ARCHIVE/summary.json" \
  --markdown-output "$ARCHIVE/results.md" \
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
