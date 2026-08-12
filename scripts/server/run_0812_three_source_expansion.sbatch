#!/usr/bin/env bash
#SBATCH --job-name=wma-3src-exp
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/share/guozhix/WMagentattack-unified-aug12/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-unified-aug12/logs/%x-%j.err

set -euo pipefail

WT=/share/guozhix/WMagentattack-unified-aug12
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
PROTOCOL=configs/0812_three_source_unified_action_protocol.json
DATASET=/share/guozhix/wmagentattack/0812/three_source_unified_action/preflight_v1/a/dataset.json
AUDIT=/share/guozhix/wmagentattack/0812/three_source_unified_action/preflight_v1/a/audit.json
ARCHIVE=/share/guozhix/wmagentattack/0812/three_source_unified_action/formal_v1

cd "$WT"
if [[ -e "$ARCHIVE" ]]; then
  echo "Refusing to overwrite frozen three-source archive: $ARCHIVE" >&2
  exit 2
fi
mkdir -p "$ARCHIVE/data" "$ARCHIVE/model" "$ARCHIVE/source_snapshot" logs
export PYTHONPATH="$WT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

failure_record() {
  status=$?
  {
    echo "failed_at=$(date --iso-8601=seconds)"
    echo "exit_status=$status"
  } > "$ARCHIVE/FAILED"
  exit "$status"
}
trap failure_record ERR

cp "$PROTOCOL" "$ARCHIVE/protocol.frozen.json"
cp "$DATASET" "$ARCHIVE/data/dataset.json"
cp "$AUDIT" "$ARCHIVE/data/audit.json"
cp \
  src/wmagentattack/hybrid_semantic_world_model.py \
  src/wmagentattack/multisource_suitability.py \
  src/wmagentattack/multisource_suitability_experiment.py \
  src/wmagentattack/three_source_unified.py \
  scripts/181_build_three_source_unified_dataset.py \
  scripts/182_train_three_source_expansion.py \
  scripts/183_summarize_three_source_expansion.py \
  scripts/server/run_0812_three_source_expansion.sbatch \
  "$ARCHIVE/source_snapshot/"

{
  echo "job_id=$SLURM_JOB_ID"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "git_head=$(git rev-parse HEAD)"
  "$PY" --version
  "$PY" -c 'import torch; print("torch=" + torch.__version__); print("cuda=" + str(torch.cuda.is_available()))'
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  echo "victim_model_calls=0"
  echo "tool_executions=0"
  echo "real_external_endpoint_calls=0"
  echo "attacks_constructed=0"
  echo "dreamer_training=0"
} > "$ARCHIVE/runtime.txt"

sha256sum \
  "$PROTOCOL" \
  "$DATASET" \
  "$AUDIT" \
  src/wmagentattack/hybrid_semantic_world_model.py \
  src/wmagentattack/multisource_suitability.py \
  src/wmagentattack/multisource_suitability_experiment.py \
  src/wmagentattack/three_source_unified.py \
  scripts/181_build_three_source_unified_dataset.py \
  scripts/182_train_three_source_expansion.py \
  scripts/183_summarize_three_source_expansion.py \
  scripts/server/run_0812_three_source_expansion.sbatch \
  tests/test_three_source_unified.py \
  tests/test_multisource_suitability.py \
  tests/test_multisource_suitability_experiment.py \
  tests/test_hybrid_semantic_world_model.py \
  > "$ARCHIVE/prerun.sha256"

"$PY" -m pytest -q \
  tests/test_three_source_unified.py \
  tests/test_multisource_suitability.py \
  tests/test_multisource_suitability_experiment.py \
  tests/test_hybrid_semantic_world_model.py \
  > "$ARCHIVE/preflight_tests.txt"

"$PY" scripts/182_train_three_source_expansion.py \
  --protocol "$PROTOCOL" \
  --dataset "$ARCHIVE/data/dataset.json" \
  --audit "$ARCHIVE/data/audit.json" \
  --output-dir "$ARCHIVE/model" \
  --device cuda \
  > "$ARCHIVE/training_stdout.json"

"$PY" scripts/183_summarize_three_source_expansion.py \
  --protocol "$PROTOCOL" \
  --dataset "$ARCHIVE/data/dataset.json" \
  --audit "$ARCHIVE/data/audit.json" \
  --predictions "$ARCHIVE/model/predictions.jsonl" \
  --run-metrics "$ARCHIVE/model/run_metrics.json" \
  --output "$ARCHIVE/summary.json" \
  --markdown "$ARCHIVE/results.md" \
  > "$ARCHIVE/summary_stdout.json"

sha256sum -c "$ARCHIVE/prerun.sha256" > "$ARCHIVE/prerun.postrun.verify.txt"
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
echo "THREE_SOURCE_EXPANSION_DONE $(date -Is)"
