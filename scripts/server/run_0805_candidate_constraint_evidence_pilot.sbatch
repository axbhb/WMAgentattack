#!/usr/bin/env bash
#SBATCH --job-name=wma-cc-data-v1
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/share/guozhix/WMagentattack-evidence-pilot-aug5/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-evidence-pilot-aug5/logs/%x-%j.err

set -euo pipefail

WT=/share/guozhix/WMagentattack-evidence-pilot-aug5
ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
PROTOCOL=configs/0805_candidate_constraint_evidence_pilot_protocol.json
RAW=/share/guozhix/wmagentattack/0729/custom_panel_v2_architecture_ablation/preflight_v2/panel_v2_architecture_dataset.json
SEMANTIC=/share/guozhix/wmagentattack/0804/autoresearch_semantic_wm_v3/stage1/rebuild_a/semantic_state_v3_dataset.json
ARCHIVE=/share/guozhix/wmagentattack/0805/candidate_constraint_evidence_pilot/fixed_v1

cd "$WT"
if [[ -e "$ARCHIVE" ]]; then
  echo "Refusing to overwrite frozen candidate-constraint archive: $ARCHIVE" >&2
  exit 2
fi
mkdir -p "$ARCHIVE/rebuild_a" "$ARCHIVE/rebuild_b" "$ARCHIVE/source_snapshot" logs
export PYTHONPATH="$WT/src:$ROOT/external/agentdojo/src:${PYTHONPATH:-}"
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

cp "$PROTOCOL" "$ARCHIVE/protocol.preregistered.json"
cp reports/0805_candidate_constraint_evidence_pilot_preregistration.md "$ARCHIVE/preregistration.md"
cp \
  src/wmagentattack/candidate_constraint_dataset.py \
  scripts/151_build_candidate_constraint_evidence_pilot.py \
  scripts/server/run_0805_candidate_constraint_evidence_pilot.sbatch \
  tests/test_candidate_constraint_dataset.py \
  "$ARCHIVE/source_snapshot/"

{
  echo "job_id=$SLURM_JOB_ID"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "git_head=$(git rev-parse HEAD)"
  "$PY" --version
  echo "victim_model_calls=0"
  echo "counterfactual_tool_executions=0"
  echo "model_training_runs=0"
  echo "attack_examples=0"
  echo "dreamer_runs=0"
} > "$ARCHIVE/runtime.txt"

sha256sum \
  "$PROTOCOL" \
  reports/0805_candidate_constraint_evidence_pilot_preregistration.md \
  src/wmagentattack/candidate_constraint_dataset.py \
  scripts/151_build_candidate_constraint_evidence_pilot.py \
  scripts/server/run_0805_candidate_constraint_evidence_pilot.sbatch \
  tests/test_candidate_constraint_dataset.py \
  > "$ARCHIVE/code.prerun.sha256"

"$PY" -m pytest -q \
  tests/test_candidate_constraint_dataset.py \
  tests/test_semantic_state_v3.py \
  tests/test_hybrid_semantic_world_model.py \
  > "$ARCHIVE/tests.txt"

"$PY" scripts/151_build_candidate_constraint_evidence_pilot.py \
  --protocol "$PROTOCOL" \
  --raw-dataset "$RAW" \
  --semantic-dataset "$SEMANTIC" \
  --output "$ARCHIVE/rebuild_a/dataset.json" \
  --audit "$ARCHIVE/rebuild_a/audit.json" \
  > "$ARCHIVE/rebuild_a/stdout.json"

"$PY" scripts/151_build_candidate_constraint_evidence_pilot.py \
  --protocol "$PROTOCOL" \
  --raw-dataset "$RAW" \
  --semantic-dataset "$SEMANTIC" \
  --output "$ARCHIVE/rebuild_b/dataset.json" \
  --audit "$ARCHIVE/rebuild_b/audit.json" \
  > "$ARCHIVE/rebuild_b/stdout.json"

cmp "$ARCHIVE/rebuild_a/dataset.json" "$ARCHIVE/rebuild_b/dataset.json"
cmp "$ARCHIVE/rebuild_a/audit.json" "$ARCHIVE/rebuild_b/audit.json"
cp "$ARCHIVE/rebuild_a/dataset.json" "$ARCHIVE/dataset.json"
cp "$ARCHIVE/rebuild_a/audit.json" "$ARCHIVE/audit.json"
{
  echo "dataset_byte_identical=true"
  echo "audit_byte_identical=true"
  sha256sum \
    "$ARCHIVE/rebuild_a/dataset.json" \
    "$ARCHIVE/rebuild_b/dataset.json" \
    "$ARCHIVE/rebuild_a/audit.json" \
    "$ARCHIVE/rebuild_b/audit.json"
} > "$ARCHIVE/rebuild_verification.txt"

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
echo "CANDIDATE_CONSTRAINT_EVIDENCE_PILOT_DONE $(date -Is)"
