#!/usr/bin/env bash
#SBATCH --job-name=wma-relational-data-v29
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/share/guozhix/WMagentattack-relational-link-aug24/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-relational-link-aug24/logs/%x-%j.err

set -euo pipefail
WT=/share/guozhix/WMagentattack-relational-link-aug24
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
PROTOCOL=configs/0824_relational_successor_data_v29_protocol.json
ROOT=/share/guozhix/wmagentattack/0824/relational_successor_data_v29/formal_v1
V17=/share/guozhix/wmagentattack/0818/paired_branch_identifiability_v17/formal_v1/paired_branch_dataset.json
V18=/share/guozhix/wmagentattack/0818/parameter_intervention_v18/formal_v1/parameter_intervention_dataset.json
V19=/share/guozhix/wmagentattack/0818/persistence_conflict_v19/formal_v2/persistence_conflict_dataset.json
HARD=/share/guozhix/wmagentattack/0821/hard_label_confirmation_v21/preflight_v1/a/dataset.json
STRUCTURED=/share/guozhix/wmagentattack/0824/successor_evidence_identifiability_v27/formal_v1/build_a/dataset.json
SUPPORT=/share/guozhix/wmagentattack/0824/explicit_atom_support_v25/formal_v1/execution_dataset.json
cd "$WT"
test ! -e "$ROOT/COMPLETE"
test "$(git rev-parse HEAD)" = "$(cat "$ROOT/GIT_HEAD")"
sha256sum -c "$ROOT/code_and_data.prerun.sha256" > "$ROOT/prerun.verify.txt"
mkdir -p "$ROOT/build_a" "$ROOT/build_b"
PYTHONPATH=src "$PY" -m pytest -q tests/test_relational_successor_contract.py > "$ROOT/tests.txt"
START=$(date +%s)
for BUILD in a b; do
  PYTHONPATH=src "$PY" scripts/269_build_relational_successor_contract_v29.py \
    --protocol "$PROTOCOL" --v17 "$V17" --v18 "$V18" --v19 "$V19" \
    --hard "$HARD" --structured "$STRUCTURED" --support-execution "$SUPPORT" \
    --base-adapters configs/0726_travel_tool_output_adapter_registry_v2.json \
    --extension-adapters configs/0729_custom_panel_v2_adapter_extension.json \
    --output-schemas configs/0824_static_tool_output_schema_v29.json \
    --output "$ROOT/build_${BUILD}/dataset.json" --audit "$ROOT/build_${BUILD}/audit.json" \
    > "$ROOT/build_${BUILD}/stdout.json"
done
PYTHONPATH=src "$PY" scripts/270_gate_relational_successor_contract_v29.py \
  --protocol "$PROTOCOL" --dataset-a "$ROOT/build_a/dataset.json" --audit-a "$ROOT/build_a/audit.json" \
  --dataset-b "$ROOT/build_b/dataset.json" --audit-b "$ROOT/build_b/audit.json" \
  --output "$ROOT/gate.json" > "$ROOT/gate_stdout.json"
END=$(date +%s)
printf 'start=%s\nend=%s\nelapsed_seconds=%s\n' "$START" "$END" "$((END-START))" > "$ROOT/runtime.txt"
find "$ROOT" -type f -not -name checksums.sha256 -not -name COMPLETE -print0 | sort -z | xargs -0 sha256sum > "$ROOT/checksums.sha256"
date -Is > "$ROOT/COMPLETE"
