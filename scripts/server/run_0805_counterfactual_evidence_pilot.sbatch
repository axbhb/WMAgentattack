#!/usr/bin/env bash
#SBATCH --job-name=wma-cf-evidence-v1
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/share/guozhix/WMagentattack-counterfactual-pilot-aug5/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-counterfactual-pilot-aug5/logs/%x-%j.err

set -euo pipefail

WT=/share/guozhix/WMagentattack-counterfactual-pilot-aug5
ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
PROTOCOL=configs/0805_counterfactual_evidence_pilot_protocol.json
MANIFEST=/share/guozhix/wmagentattack/0805/counterfactual_evidence_execution_pilot/manifest_preflight_v1/a/manifest.json
MANIFEST_AUDIT=/share/guozhix/wmagentattack/0805/counterfactual_evidence_execution_pilot/manifest_preflight_v1/a/audit.json
RAW=/share/guozhix/wmagentattack/0729/custom_panel_v2_architecture_ablation/preflight_v2/panel_v2_architecture_dataset.json
SEMANTIC=/share/guozhix/wmagentattack/0804/autoresearch_semantic_wm_v3/stage1/rebuild_a/semantic_state_v3_dataset.json
ADAPTER=configs/0729_custom_panel_v2_adapter_extension.json
ARCHIVE=/share/guozhix/wmagentattack/0805/counterfactual_evidence_execution_pilot/fixed_v1

cd "$WT"
if [[ -e "$ARCHIVE" ]]; then
  echo "Refusing to overwrite frozen counterfactual archive: $ARCHIVE" >&2
  exit 2
fi
mkdir -p "$ARCHIVE/source_snapshot" logs
export PYTHONPATH="$WT/src:$ROOT/external/agentdojo/src"
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
cp "$MANIFEST" "$ARCHIVE/manifest.frozen.json"
cp "$MANIFEST_AUDIT" "$ARCHIVE/manifest_preflight_audit.json"
cp reports/0805_counterfactual_evidence_pilot_preregistration.md "$ARCHIVE/preregistration.md"
cp \
  src/wmagentattack/counterfactual_evidence.py \
  src/wmagentattack/counterfactual_execution.py \
  scripts/152_build_counterfactual_evidence_manifest.py \
  scripts/153_run_counterfactual_evidence_pilot.py \
  scripts/server/run_0805_counterfactual_evidence_pilot.sbatch \
  tests/test_counterfactual_evidence.py \
  "$ARCHIVE/source_snapshot/"

{
  echo "job_id=$SLURM_JOB_ID"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "git_head=$(git rev-parse HEAD)"
  "$PY" --version
  echo "victim_model_calls=0"
  echo "counterfactual_tool_executions=48"
  echo "observed_prefix_replay_tool_executions=36"
  echo "total_sandbox_tool_executions=84"
  echo "attack_examples=0"
  echo "model_training_runs=0"
  echo "dreamer_runs=0"
} > "$ARCHIVE/runtime.txt"

sha256sum \
  "$PROTOCOL" "$MANIFEST" "$MANIFEST_AUDIT" "$ADAPTER" \
  src/wmagentattack/counterfactual_evidence.py \
  src/wmagentattack/counterfactual_execution.py \
  scripts/152_build_counterfactual_evidence_manifest.py \
  scripts/153_run_counterfactual_evidence_pilot.py \
  scripts/server/run_0805_counterfactual_evidence_pilot.sbatch \
  tests/test_counterfactual_evidence.py \
  > "$ARCHIVE/code_and_manifest.prerun.sha256"

"$PY" -m pytest -q \
  tests/test_counterfactual_evidence.py \
  tests/test_candidate_constraint_dataset.py \
  tests/test_semantic_state_v3.py \
  tests/test_hybrid_semantic_world_model.py \
  > "$ARCHIVE/tests.txt"

"$PY" scripts/153_run_counterfactual_evidence_pilot.py \
  --protocol "$PROTOCOL" \
  --manifest "$MANIFEST" \
  --raw-dataset "$RAW" \
  --semantic-dataset "$SEMANTIC" \
  --adapter-extension "$ADAPTER" \
  --output "$ARCHIVE/counterfactual_delta_dataset.json" \
  --audit "$ARCHIVE/audit.json" \
  > "$ARCHIVE/stdout.json"

sha256sum -c "$ARCHIVE/code_and_manifest.prerun.sha256" \
  > "$ARCHIVE/code_and_manifest.postrun.verify.txt"
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
echo "COUNTERFACTUAL_EVIDENCE_PILOT_DONE $(date -Is)"
