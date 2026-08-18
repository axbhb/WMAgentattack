#!/usr/bin/env bash
#SBATCH --job-name=wma-branch-v17
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/share/guozhix/WMagentattack-branching-identifiability-aug18/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-branching-identifiability-aug18/logs/%x-%j.err

set -euo pipefail

WT=/share/guozhix/WMagentattack-branching-identifiability-aug18
ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
PROTOCOL=configs/0818_paired_branch_identifiability_v17_protocol.json
MANIFEST=/share/guozhix/wmagentattack/0818/paired_branch_identifiability_v17/manifest_preflight_v1/a/manifest.json
MANIFEST_AUDIT=/share/guozhix/wmagentattack/0818/paired_branch_identifiability_v17/manifest_preflight_v1/a/audit.json
RAW=/share/guozhix/wmagentattack/0729/custom_panel_v2_architecture_ablation/preflight_v2/panel_v2_architecture_dataset.json
SEMANTIC=/share/guozhix/wmagentattack/0804/autoresearch_semantic_wm_v3/stage1/rebuild_a/semantic_state_v3_dataset.json
ADAPTER=configs/0729_custom_panel_v2_adapter_extension.json
ADAPTER_REPAIR=configs/0805_counterfactual_adapter_repair_v2.json
ARCHIVE=/share/guozhix/wmagentattack/0818/paired_branch_identifiability_v17/formal_v1

cd "$WT"
if [[ -e "$ARCHIVE" ]]; then
  echo "Refusing to overwrite frozen v17 archive: $ARCHIVE" >&2
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
cp reports/0818_paired_branch_identifiability_v17_preregistration.md "$ARCHIVE/preregistration.md"
cp \
  src/wmagentattack/branching_identifiability.py \
  src/wmagentattack/counterfactual_execution.py \
  scripts/236_build_paired_branch_manifest_v17.py \
  scripts/237_run_paired_branch_pilot_v17.py \
  scripts/server/run_0818_paired_branch_identifiability_v17.sbatch \
  tests/test_branching_identifiability.py \
  "$ARCHIVE/source_snapshot/"

{
  echo "job_id=$SLURM_JOB_ID"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "git_head=$(git rev-parse HEAD)"
  "$PY" --version
  echo "clean_only=true"
  echo "victim_model_calls=0"
  echo "unique_bound_queries=48"
  echo "counterfactual_tool_executions=96"
  echo "observed_prefix_replay_tool_executions=152"
  echo "total_sandbox_tool_executions=248"
  echo "attack_examples=0"
  echo "model_training_runs=0"
  echo "dreamer_runs=0"
  echo "real_external_endpoint_calls=0"
} > "$ARCHIVE/runtime.txt"

sha256sum \
  "$PROTOCOL" "$MANIFEST" "$MANIFEST_AUDIT" "$ADAPTER" "$ADAPTER_REPAIR" \
  src/wmagentattack/branching_identifiability.py \
  src/wmagentattack/counterfactual_execution.py \
  scripts/236_build_paired_branch_manifest_v17.py \
  scripts/237_run_paired_branch_pilot_v17.py \
  scripts/server/run_0818_paired_branch_identifiability_v17.sbatch \
  tests/test_branching_identifiability.py \
  > "$ARCHIVE/code_and_manifest.prerun.sha256"

"$PY" -m pytest -q \
  tests/test_branching_identifiability.py \
  tests/test_counterfactual_evidence.py \
  tests/test_semantic_state_v3.py \
  > "$ARCHIVE/tests.txt"

"$PY" scripts/237_run_paired_branch_pilot_v17.py \
  --protocol "$PROTOCOL" \
  --manifest "$MANIFEST" \
  --raw-dataset "$RAW" \
  --semantic-dataset "$SEMANTIC" \
  --adapter-extension "$ADAPTER" \
  --adapter-repair "$ADAPTER_REPAIR" \
  --output "$ARCHIVE/paired_branch_dataset.json" \
  --audit "$ARCHIVE/gate.json" \
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
echo "PAIRED_BRANCH_IDENTIFIABILITY_V17_DONE $(date -Is)"
