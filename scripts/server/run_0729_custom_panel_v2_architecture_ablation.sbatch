#!/usr/bin/env bash
#SBATCH --job-name=wma-panel-v2-arch
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%j.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
PROTOCOL=configs/0729_custom_panel_v2_architecture_ablation_protocol.json
SOURCE=/share/guozhix/wmagentattack/0729/custom_panel_v2_architecture_ablation/preflight_v1
ARCHIVE=/share/guozhix/wmagentattack/0729/custom_panel_v2_architecture_ablation/fixed_v1
DATASET="$ARCHIVE/data"
MODEL="$ARCHIVE/model"

cd "$ROOT"
if [[ -e "$ARCHIVE" ]]; then
  echo "Refusing to overwrite frozen archive: $ARCHIVE" >&2
  exit 2
fi
mkdir -p "$ARCHIVE" "$DATASET" "$MODEL" "$ARCHIVE/source_snapshot" logs
export PYTHONPATH="$ROOT/src:$ROOT/external/agentdojo/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
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

cp "$PROTOCOL" "$ARCHIVE/protocol.preregistered.json"
cp reports/0729_custom_panel_v2_architecture_ablation_preregistration.md "$ARCHIVE/preregistration.md"
cp configs/0729_custom_panel_v2_adapter_extension.json "$ARCHIVE/adapter_extension.json"
cp "$SOURCE/panel_v2_architecture_dataset.json" "$DATASET/panel_v2_architecture_dataset.json"
cp "$SOURCE/dataset_audit.json" "$DATASET/dataset_audit.json"
cp \
  src/wmagentattack/structured_ledger_v2.py \
  src/wmagentattack/factorized_evaluator_v2.py \
  src/wmagentattack/panel_v2_architecture_probe.py \
  scripts/142_build_custom_panel_v2_architecture_dataset.py \
  scripts/143_train_custom_panel_v2_architecture_ablation.py \
  scripts/144_summarize_custom_panel_v2_architecture_ablation.py \
  scripts/145_audit_custom_panel_v2_architecture_preflight.py \
  "$ARCHIVE/source_snapshot/"

{
  echo "job_id=$SLURM_JOB_ID"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "git_head=$(git rev-parse HEAD)"
  "$PY" --version
  echo "victim_model_calls=0"
  echo "attacks_constructed=0"
  echo "completion_reporting_training=0"
  echo "dreamer_training=0"
} > "$ARCHIVE/runtime.txt"

sha256sum \
  "$PROTOCOL" \
  configs/0729_custom_panel_v2_adapter_extension.json \
  src/wmagentattack/structured_ledger_v2.py \
  src/wmagentattack/factorized_evaluator_v2.py \
  src/wmagentattack/panel_v2_architecture_probe.py \
  scripts/142_build_custom_panel_v2_architecture_dataset.py \
  scripts/143_train_custom_panel_v2_architecture_ablation.py \
  scripts/144_summarize_custom_panel_v2_architecture_ablation.py \
  tests/test_structured_ledger_v2.py \
  tests/test_factorized_evaluator_v2.py \
  tests/test_panel_v2_architecture_probe.py \
  tests/test_custom_panel_v2_architecture_summary.py \
  > "$ARCHIVE/code.prerun.sha256"

"$PY" -m pytest -q \
  tests/test_structured_ledger_v2.py \
  tests/test_factorized_evaluator_v2.py \
  tests/test_panel_v2_architecture_probe.py \
  tests/test_custom_panel_v2_architecture_summary.py \
  > "$ARCHIVE/preflight_tests.txt"

"$PY" scripts/145_audit_custom_panel_v2_architecture_preflight.py \
  --protocol "$PROTOCOL" \
  --dataset "$DATASET/panel_v2_architecture_dataset.json" \
  --dataset-audit "$DATASET/dataset_audit.json" \
  --output "$ARCHIVE/preflight.json" \
  > "$ARCHIVE/preflight_stdout.json"

"$PY" scripts/143_train_custom_panel_v2_architecture_ablation.py \
  --protocol "$PROTOCOL" \
  --dataset "$DATASET/panel_v2_architecture_dataset.json" \
  --output-dir "$MODEL" \
  --device cpu \
  > "$ARCHIVE/training_stdout.jsonl"

"$PY" scripts/144_summarize_custom_panel_v2_architecture_ablation.py \
  --protocol "$PROTOCOL" \
  --run-metrics "$MODEL/run_metrics.json" \
  --predictions "$MODEL/predictions.jsonl" \
  --output "$ARCHIVE/summary.json" \
  --report "$ARCHIVE/results.md" \
  > "$ARCHIVE/summary_stdout.json"

sha256sum -c "$ARCHIVE/code.prerun.sha256" > "$ARCHIVE/code.postrun.verify.txt"
{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "runtime_failures=0"
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
echo "CUSTOM_PANEL_V2_ARCHITECTURE_ABLATION_DONE $(date -Is)"
