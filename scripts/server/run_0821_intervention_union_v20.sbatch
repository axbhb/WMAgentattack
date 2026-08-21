#!/usr/bin/env bash
#SBATCH --job-name=wma-intervention-v20
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/share/guozhix/WMagentattack-intervention-union-aug21/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-intervention-union-aug21/logs/%x-%j.err

set -euo pipefail

WT=/share/guozhix/WMagentattack-intervention-union-aug21
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
PROTOCOL=configs/0821_intervention_union_v20_protocol.json
DATA=/share/guozhix/wmagentattack/0821/intervention_union_v20/preflight_v1/a/dataset.json
AUDIT=/share/guozhix/wmagentattack/0821/intervention_union_v20/preflight_v1/a/audit.json
ARCHIVE=/share/guozhix/wmagentattack/0821/intervention_union_v20/formal_v1

cd "$WT"
if [[ -e "$ARCHIVE" ]]; then
  echo "Refusing to overwrite frozen v20 archive: $ARCHIVE" >&2
  exit 2
fi
mkdir -p "$ARCHIVE/results" "$ARCHIVE/source_snapshot" logs
export PYTHONPATH="$WT/src:/share/guozhix/WMagentattack/external/agentdojo/src"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

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
cp "$AUDIT" "$ARCHIVE/union_audit.frozen.json"
cp reports/0821_intervention_union_v20_preregistration.md "$ARCHIVE/preregistration.md"
cp \
  src/wmagentattack/intervention_union.py \
  src/wmagentattack/intervention_modular_world_model.py \
  scripts/242_build_intervention_union_v20.py \
  scripts/243_train_intervention_models_v20.py \
  scripts/244_gate_intervention_models_v20.py \
  scripts/server/run_0821_intervention_union_v20.sbatch \
  tests/test_intervention_union.py \
  tests/test_intervention_modular_world_model.py \
  "$ARCHIVE/source_snapshot/"

{
  echo "job_id=$SLURM_JOB_ID"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "git_head=$(git rev-parse HEAD)"
  "$PY" --version
  echo "clean_only=true"
  echo "model_fits=27"
  echo "victim_model_calls=0"
  echo "attack_examples=0"
  echo "dreamer_runs=0"
  echo "real_external_endpoint_calls=0"
} > "$ARCHIVE/runtime.txt"

sha256sum \
  "$PROTOCOL" "$DATA" "$AUDIT" \
  src/wmagentattack/intervention_union.py \
  src/wmagentattack/intervention_modular_world_model.py \
  scripts/242_build_intervention_union_v20.py \
  scripts/243_train_intervention_models_v20.py \
  scripts/244_gate_intervention_models_v20.py \
  scripts/server/run_0821_intervention_union_v20.sbatch \
  tests/test_intervention_union.py \
  tests/test_intervention_modular_world_model.py \
  > "$ARCHIVE/code_and_data.prerun.sha256"

"$PY" -m pytest -q \
  tests/test_intervention_union.py \
  tests/test_intervention_modular_world_model.py \
  tests/test_semantic_state_v3.py \
  > "$ARCHIVE/tests.txt"

"$PY" scripts/243_train_intervention_models_v20.py \
  --protocol "$PROTOCOL" \
  --dataset "$DATA" \
  --output-dir "$ARCHIVE/results" \
  > "$ARCHIVE/train_stdout.json"

"$PY" scripts/244_gate_intervention_models_v20.py \
  --protocol "$PROTOCOL" \
  --metrics "$ARCHIVE/results/run_metrics.json" \
  --output "$ARCHIVE/results/gate.json" \
  > "$ARCHIVE/gate_stdout.json"

sha256sum -c "$ARCHIVE/code_and_data.prerun.sha256" \
  > "$ARCHIVE/code_and_data.postrun.verify.txt"
{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "runtime_failures=0"
} >> "$ARCHIVE/runtime.txt"
find "$ARCHIVE" -type f \
  -not -name checksums.sha256 \
  -not -name COMPLETE \
  -not -name FAILED \
  -print0 | sort -z | xargs -0 sha256sum > "$ARCHIVE/checksums.sha256"
date --iso-8601=seconds > "$ARCHIVE/COMPLETE"
trap - ERR
echo "INTERVENTION_UNION_V20_DONE $(date -Is)"
