#!/usr/bin/env bash
#SBATCH --job-name=wma-persist-v19
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/share/guozhix/WMagentattack-persistence-conflict-aug18/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-persistence-conflict-aug18/logs/%x-%j.err

set -euo pipefail

WT=/share/guozhix/WMagentattack-persistence-conflict-aug18
ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
PROTOCOL=configs/0818_persistence_conflict_v19_protocol.json
MANIFEST=/share/guozhix/wmagentattack/0818/persistence_conflict_v19/manifest_preflight_v2/a/manifest.json
MANIFEST_AUDIT=/share/guozhix/wmagentattack/0818/persistence_conflict_v19/manifest_preflight_v2/a/audit.json
RAW=/share/guozhix/wmagentattack/0729/custom_panel_v2_architecture_ablation/preflight_v2/panel_v2_architecture_dataset.json
SEMANTIC=/share/guozhix/wmagentattack/0804/autoresearch_semantic_wm_v3/stage1/rebuild_a/semantic_state_v3_dataset.json
ADAPTER=configs/0729_custom_panel_v2_adapter_extension.json
ADAPTER_REPAIR=configs/0805_counterfactual_adapter_repair_v2.json
PERSISTENCE_ADAPTER=configs/0818_persistence_adapter_extension_v19.json
ARCHIVE=/share/guozhix/wmagentattack/0818/persistence_conflict_v19/formal_v2

cd "$WT"
if [[ -e "$ARCHIVE" ]]; then
  echo "Refusing to overwrite frozen v19 archive: $ARCHIVE" >&2
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
cp reports/0818_persistence_conflict_v19_preregistration.md "$ARCHIVE/preregistration.md"
cp \
  configs/0818_persistence_adapter_extension_v19.json \
  src/wmagentattack/persistence_conflict.py \
  scripts/240_build_persistence_conflict_v19.py \
  scripts/241_run_persistence_conflict_v19.py \
  scripts/server/run_0818_persistence_conflict_v19.sbatch \
  tests/test_persistence_conflict.py \
  "$ARCHIVE/source_snapshot/"

{
  echo "job_id=$SLURM_JOB_ID"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "git_head=$(git rev-parse HEAD)"
  "$PY" --version
  echo "clean_only=true"
  echo "sequence_pairs=12"
  echo "canonical_sequences=24"
  echo "sequence_step_executions=144"
  echo "observed_prefix_replay_tool_executions=76"
  echo "total_sandbox_tool_executions=220"
  echo "victim_model_calls=0"
  echo "attack_examples=0"
  echo "model_training_runs=0"
  echo "dreamer_runs=0"
  echo "real_external_endpoint_calls=0"
} > "$ARCHIVE/runtime.txt"

sha256sum \
  "$PROTOCOL" "$MANIFEST" "$MANIFEST_AUDIT" "$ADAPTER" "$ADAPTER_REPAIR" "$PERSISTENCE_ADAPTER" \
  src/wmagentattack/persistence_conflict.py \
  scripts/240_build_persistence_conflict_v19.py \
  scripts/241_run_persistence_conflict_v19.py \
  scripts/server/run_0818_persistence_conflict_v19.sbatch \
  tests/test_persistence_conflict.py \
  > "$ARCHIVE/code_and_manifest.prerun.sha256"

"$PY" -m pytest -q \
  tests/test_persistence_conflict.py \
  tests/test_counterfactual_evidence.py \
  tests/test_semantic_state_v3.py \
  > "$ARCHIVE/tests.txt"

"$PY" scripts/241_run_persistence_conflict_v19.py \
  --protocol "$PROTOCOL" \
  --manifest "$MANIFEST" \
  --raw-dataset "$RAW" \
  --semantic-dataset "$SEMANTIC" \
  --adapter-extension "$ADAPTER" \
  --adapter-repair "$ADAPTER_REPAIR" \
  --persistence-adapter "$PERSISTENCE_ADAPTER" \
  --output "$ARCHIVE/persistence_conflict_dataset.json" \
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
echo "PERSISTENCE_CONFLICT_V19_DONE $(date -Is)"
