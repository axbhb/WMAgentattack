#!/usr/bin/env bash
#SBATCH --job-name=wma-support-v25
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/share/guozhix/WMagentattack-explicit-support-aug24/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-explicit-support-aug24/logs/%x-%j.err

set -euo pipefail

WT=/share/guozhix/WMagentattack-explicit-support-aug24
ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
PROTOCOL=configs/0824_explicit_atom_support_v25_protocol.json
MANIFEST=/share/guozhix/wmagentattack/0824/explicit_atom_support_v25/preflight_v1/a/manifest.json
MANIFEST_AUDIT=/share/guozhix/wmagentattack/0824/explicit_atom_support_v25/preflight_v1/a/audit.json
RAW=/share/guozhix/wmagentattack/0729/custom_panel_v2_architecture_ablation/preflight_v2/panel_v2_architecture_dataset.json
SEMANTIC=/share/guozhix/wmagentattack/0804/autoresearch_semantic_wm_v3/stage1/rebuild_a/semantic_state_v3_dataset.json
HARD=/share/guozhix/wmagentattack/0821/hard_label_confirmation_v21/preflight_v1/a/dataset.json
ADAPTER=configs/0729_custom_panel_v2_adapter_extension.json
ADAPTER_REPAIR=configs/0805_counterfactual_adapter_repair_v2.json
ARCHIVE=/share/guozhix/wmagentattack/0824/explicit_atom_support_v25/formal_v1

cd "$WT"
if [[ -e "$ARCHIVE" ]]; then
  echo "Refusing to overwrite frozen v25 archive: $ARCHIVE" >&2
  exit 2
fi
mkdir -p "$ARCHIVE/source_snapshot" logs
export PYTHONPATH="$WT/src:$ROOT/external/agentdojo/src"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

failure_record() {
  status=$?
  { echo "failed_at=$(date --iso-8601=seconds)"; echo "exit_status=$status"; } > "$ARCHIVE/FAILED"
  exit "$status"
}
trap failure_record ERR

cp "$PROTOCOL" "$ARCHIVE/protocol.frozen.json"
cp "$MANIFEST" "$ARCHIVE/manifest.frozen.json"
cp "$MANIFEST_AUDIT" "$ARCHIVE/manifest_preflight_audit.json"
cp reports/0824_explicit_atom_support_v25_preregistration.md "$ARCHIVE/preregistration.md"
cp src/wmagentattack/explicit_support_panel.py scripts/261_build_explicit_support_manifest_v25.py scripts/262_run_explicit_support_data_gate_v25.py scripts/server/run_0824_explicit_atom_support_v25.sbatch tests/test_explicit_support_panel.py "$ARCHIVE/source_snapshot/"

{
  echo "job_id=$SLURM_JOB_ID"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "git_head=$(git rev-parse HEAD)"
  "$PY" --version
  echo "clean_only=true"
  echo "victim_model_calls=0"
  echo "attack_examples=0"
  echo "model_training_runs=0"
  echo "dreamer_runs=0"
  echo "real_external_endpoint_calls=0"
} > "$ARCHIVE/runtime.txt"

sha256sum "$PROTOCOL" "$MANIFEST" "$MANIFEST_AUDIT" "$ADAPTER" "$ADAPTER_REPAIR" src/wmagentattack/explicit_support_panel.py scripts/261_build_explicit_support_manifest_v25.py scripts/262_run_explicit_support_data_gate_v25.py scripts/server/run_0824_explicit_atom_support_v25.sbatch tests/test_explicit_support_panel.py > "$ARCHIVE/code_and_manifest.prerun.sha256"

"$PY" -m pytest -q tests/test_explicit_support_panel.py tests/test_branching_identifiability.py tests/test_counterfactual_evidence.py tests/test_semantic_state_v3.py > "$ARCHIVE/tests.txt"

"$PY" scripts/262_run_explicit_support_data_gate_v25.py --protocol "$PROTOCOL" --manifest "$MANIFEST" --raw-dataset "$RAW" --semantic-dataset "$SEMANTIC" --hard-dataset "$HARD" --adapter-extension "$ADAPTER" --adapter-repair "$ADAPTER_REPAIR" --execution-output "$ARCHIVE/execution_dataset.json" --output "$ARCHIVE/atom_support_dataset.json" --gate "$ARCHIVE/gate.json" > "$ARCHIVE/stdout.json"

sha256sum -c "$ARCHIVE/code_and_manifest.prerun.sha256" > "$ARCHIVE/code_and_manifest.postrun.verify.txt"
{ echo "finished_at=$(date --iso-8601=seconds)"; echo "runtime_failures=0"; } >> "$ARCHIVE/runtime.txt"
find "$ARCHIVE" -type f -not -name checksums.sha256 -not -name COMPLETE -not -name FAILED -print0 | sort -z | xargs -0 sha256sum > "$ARCHIVE/checksums.sha256"
date --iso-8601=seconds > "$ARCHIVE/COMPLETE"
trap - ERR
echo "EXPLICIT_ATOM_SUPPORT_V25_DONE $(date -Is)"
