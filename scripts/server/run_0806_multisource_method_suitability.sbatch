#!/usr/bin/env bash
#SBATCH --job-name=wma-ms-suit
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/share/guozhix/WMagentattack-multisource-suitability-aug5/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-multisource-suitability-aug5/logs/%x-%j.err

set -euo pipefail

WT=/share/guozhix/WMagentattack-multisource-suitability-aug5
ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
PROTOCOL=configs/0806_multisource_method_suitability_protocol.json
DATASET=/share/guozhix/wmagentattack/0806/multisource_method_suitability/preflight_v2/a/dataset.json
AUDIT=/share/guozhix/wmagentattack/0806/multisource_method_suitability/preflight_v2/a/audit.json
ARCHIVE=/share/guozhix/wmagentattack/0806/multisource_method_suitability/fixed_v1

cd "$WT"
if [[ -e "$ARCHIVE" ]]; then
  echo "Refusing to overwrite frozen suitability archive: $ARCHIVE" >&2
  exit 2
fi
mkdir -p "$ARCHIVE/data" "$ARCHIVE/model" "$ARCHIVE/source_snapshot" logs
export PYTHONPATH="$WT/src:$ROOT/external/agentdojo/src:${PYTHONPATH:-}"
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
cp reports/0806_multisource_method_suitability_preregistration.md "$ARCHIVE/preregistration.md"
cp "$DATASET" "$ARCHIVE/data/dataset.json"
cp "$AUDIT" "$ARCHIVE/data/audit.json"
cp \
  src/wmagentattack/hybrid_semantic_world_model.py \
  src/wmagentattack/multisource_suitability.py \
  src/wmagentattack/multisource_suitability_experiment.py \
  scripts/160_train_multisource_suitability.py \
  scripts/161_summarize_multisource_suitability.py \
  scripts/server/run_0806_multisource_method_suitability.sbatch \
  "$ARCHIVE/source_snapshot/"

{
  echo "job_id=$SLURM_JOB_ID"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "git_head=$(git rev-parse HEAD)"
  "$PY" --version
  "$PY" -c 'import torch, sklearn; print("torch=" + torch.__version__); print("sklearn=" + sklearn.__version__); print("cuda=" + str(torch.cuda.is_available()))'
  echo "victim_model_calls=0"
  echo "tool_executions=0"
  echo "attacks_constructed=0"
  echo "dreamer_training=0"
} > "$ARCHIVE/runtime.txt"

sha256sum \
  "$PROTOCOL" \
  reports/0806_multisource_method_suitability_preregistration.md \
  src/wmagentattack/hybrid_semantic_world_model.py \
  src/wmagentattack/multisource_suitability.py \
  src/wmagentattack/multisource_suitability_experiment.py \
  scripts/160_train_multisource_suitability.py \
  scripts/161_summarize_multisource_suitability.py \
  scripts/server/run_0806_multisource_method_suitability.sbatch \
  tests/test_multisource_suitability.py \
  tests/test_multisource_suitability_experiment.py \
  tests/test_hybrid_semantic_world_model.py \
  > "$ARCHIVE/code.prerun.sha256"

"$PY" -m pytest -q \
  tests/test_multisource_suitability.py \
  tests/test_multisource_suitability_experiment.py \
  tests/test_hybrid_semantic_world_model.py \
  > "$ARCHIVE/preflight_tests.txt"

"$PY" scripts/160_train_multisource_suitability.py \
  --protocol "$PROTOCOL" \
  --dataset "$ARCHIVE/data/dataset.json" \
  --audit "$ARCHIVE/data/audit.json" \
  --output-dir "$ARCHIVE/model" \
  --device cuda \
  > "$ARCHIVE/training_stdout.json"

"$PY" scripts/161_summarize_multisource_suitability.py \
  --protocol "$PROTOCOL" \
  --audit "$ARCHIVE/data/audit.json" \
  --run-metrics "$ARCHIVE/model/run_metrics.json" \
  --predictions "$ARCHIVE/model/predictions.jsonl" \
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
echo "MULTISOURCE_METHOD_SUITABILITY_DONE $(date -Is)"
