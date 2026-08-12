#!/usr/bin/env bash
#SBATCH --job-name=wma-onto-cpu
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/share/guozhix/WMagentattack-repair-aug12/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-repair-aug12/logs/%x-%j.err

set -euo pipefail

WT=/share/guozhix/WMagentattack-repair-aug12
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
PROTOCOL=configs/0813_shared_action_ontology_protocol.json
DATASET=/share/guozhix/wmagentattack/0813/action_ontology/preflight_v3/a/dataset.json
AUDIT=/share/guozhix/wmagentattack/0813/action_ontology/preflight_v3/a/audit.json
PARENT=/share/guozhix/wmagentattack/0812/three_source_unified_action/formal_v1/model/predictions.jsonl
ARCHIVE=/share/guozhix/wmagentattack/0813/action_ontology/formal_v3

cd "$WT"
if [[ -e "$ARCHIVE" ]]; then
  echo "Refusing to overwrite frozen action-ontology archive: $ARCHIVE" >&2
  exit 2
fi
mkdir -p "$ARCHIVE/data" "$ARCHIVE/model" "$ARCHIVE/source_snapshot" logs
export PYTHONPATH="$WT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

failure_record() {
  status=$?
  { echo "failed_at=$(date --iso-8601=seconds)"; echo "exit_status=$status"; } > "$ARCHIVE/FAILED"
  exit "$status"
}
trap failure_record ERR

cp "$PROTOCOL" "$ARCHIVE/protocol.frozen.json"
cp "$DATASET" "$ARCHIVE/data/dataset.json"
cp "$AUDIT" "$ARCHIVE/data/audit.json"
cp \
  src/wmagentattack/shared_action_ontology.py \
  scripts/184_build_shared_action_ontology.py \
  scripts/185_train_shared_action_ontology.py \
  scripts/186_summarize_shared_action_ontology.py \
  scripts/server/run_0813_shared_action_ontology_cpu.sbatch \
  "$ARCHIVE/source_snapshot/"

{
  echo "job_id=$SLURM_JOB_ID"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "git_head=$(git rev-parse HEAD)"
  "$PY" --version
  "$PY" -c 'import torch; print("torch=" + torch.__version__); print("training_device=cpu"); print("torch_threads=" + str(torch.get_num_threads()))'
  echo "cancelled_pending_gpu_job=6732"
  echo "new_llm_calls=0"
  echo "new_tool_executions=0"
  echo "real_external_endpoint_calls=0"
  echo "attack_generation=0"
  echo "dreamer_runs=0"
} > "$ARCHIVE/runtime.txt"

sha256sum \
  "$PROTOCOL" "$DATASET" "$AUDIT" "$PARENT" \
  src/wmagentattack/shared_action_ontology.py \
  scripts/184_build_shared_action_ontology.py \
  scripts/185_train_shared_action_ontology.py \
  scripts/186_summarize_shared_action_ontology.py \
  scripts/server/run_0813_shared_action_ontology_cpu.sbatch \
  tests/test_shared_action_ontology.py \
  tests/test_three_source_unified.py \
  tests/test_hybrid_semantic_world_model.py \
  > "$ARCHIVE/prerun.sha256"

"$PY" -m pytest -q \
  tests/test_shared_action_ontology.py \
  tests/test_three_source_unified.py \
  tests/test_hybrid_semantic_world_model.py \
  > "$ARCHIVE/preflight_tests.txt"

"$PY" scripts/185_train_shared_action_ontology.py \
  --protocol "$PROTOCOL" --dataset "$DATASET" --audit "$AUDIT" \
  --output-dir "$ARCHIVE/model" --device cpu \
  > "$ARCHIVE/training_stdout.json"

"$PY" scripts/186_summarize_shared_action_ontology.py \
  --protocol "$PROTOCOL" --audit "$AUDIT" \
  --run-metrics "$ARCHIVE/model/run_metrics.json" \
  --predictions "$ARCHIVE/model/predictions.jsonl" \
  --parent-predictions "$PARENT" \
  --output "$ARCHIVE/summary.json" --markdown "$ARCHIVE/results.md" \
  > "$ARCHIVE/summary_stdout.json"

sha256sum -c "$ARCHIVE/prerun.sha256" > "$ARCHIVE/prerun.postrun.verify.txt"
{ echo "finished_at=$(date --iso-8601=seconds)"; echo "runtime_failures=0"; } >> "$ARCHIVE/runtime.txt"
find "$ARCHIVE" -type f -not -name checksums.sha256 -not -name COMPLETE -not -name FAILED -print0 \
  | sort -z | xargs -0 sha256sum > "$ARCHIVE/checksums.sha256"
date --iso-8601=seconds > "$ARCHIVE/COMPLETE"
trap - ERR
echo "SHARED_ACTION_ONTOLOGY_CPU_DONE $(date -Is)"
