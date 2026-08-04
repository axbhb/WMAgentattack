#!/usr/bin/env bash
#SBATCH --job-name=wma-clean-evidence
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%j.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
PROTOCOL=configs/0725_clean_evidence_ledger_ablation_protocol.json
ARCHIVE=/share/guozhix/wmagentattack/0725/clean_evidence_ledger_ablation/fixed_v1
DATASET="$ARCHIVE/data"
MODEL="$ARCHIVE/model"

cd "$ROOT"
if [[ -e "$ARCHIVE" ]]; then
  echo "Refusing to overwrite frozen archive: $ARCHIVE" >&2
  exit 2
fi
mkdir -p "$ARCHIVE" "$DATASET" "$MODEL" logs
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

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

cp "$PROTOCOL" "$ARCHIVE/protocol.preregistered.json"
sha256sum \
  "$PROTOCOL" \
  src/wmagentattack/clean_state_instrumentation.py \
  src/wmagentattack/trace_execution_pairing.py \
  src/wmagentattack/evidence_ledger.py \
  src/wmagentattack/clean_evidence_probe.py \
  scripts/121_build_clean_evidence_ledger_dataset.py \
  scripts/122_train_clean_evidence_ablation.py \
  scripts/123_summarize_clean_evidence_ablation.py \
  tests/test_evidence_ledger.py \
  tests/test_clean_evidence_dataset_builder.py \
  tests/test_clean_evidence_probe.py \
  tests/test_clean_evidence_ablation_summary.py \
  > "$ARCHIVE/code.prerun.sha256"

"$PY" -m pytest -q \
  tests/test_evidence_ledger.py \
  tests/test_clean_evidence_dataset_builder.py \
  tests/test_clean_evidence_probe.py \
  tests/test_clean_evidence_ablation_summary.py \
  > "$ARCHIVE/preflight_tests.txt"

"$PY" scripts/121_build_clean_evidence_ledger_dataset.py \
  --protocol "$PROTOCOL" \
  --output-dir "$DATASET" \
  > "$ARCHIVE/dataset_stdout.json"

"$PY" scripts/122_train_clean_evidence_ablation.py \
  --protocol "$PROTOCOL" \
  --dataset "$DATASET/episodes.jsonl" \
  --output-dir "$MODEL" \
  --device cuda \
  > "$ARCHIVE/training_stdout.jsonl"

"$PY" scripts/123_summarize_clean_evidence_ablation.py \
  --protocol "$PROTOCOL" \
  --audit "$DATASET/audit.json" \
  --training-manifest "$MODEL/training_manifest.json" \
  --predictions "$MODEL/oof_predictions.jsonl" \
  --output-json "$ARCHIVE/summary.json" \
  --output-markdown "$ARCHIVE/results.md" \
  > "$ARCHIVE/summary_stdout.json"

"$PY" -m pytest -q > "$ARCHIVE/full_tests.txt"
sha256sum -c "$ARCHIVE/code.prerun.sha256" > "$ARCHIVE/code.postrun.verify.txt"
{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "runtime_failures=0"
  echo "attacks_constructed=0"
  echo "victim_model_calls=0"
  echo "external_endpoints=0"
  echo "dreamer_training=0"
} >> "$ARCHIVE/runtime.txt"
find "$ARCHIVE" -type f \
  -not -name checksums.sha256 \
  -not -name COMPLETE \
  -not -name FAILED \
  -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$ARCHIVE/checksums.sha256"
date --iso-8601=seconds > "$ARCHIVE/COMPLETE"
trap - ERR
