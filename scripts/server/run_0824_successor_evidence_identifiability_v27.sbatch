#!/usr/bin/env bash
#SBATCH --job-name=wma-successor-evidence-v27
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/share/guozhix/WMagentattack-successor-evidence-aug24/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-successor-evidence-aug24/logs/%x-%j.err

set -euo pipefail
WT=/share/guozhix/WMagentattack-successor-evidence-aug24
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
PROTOCOL=configs/0824_successor_evidence_identifiability_v27_protocol.json
ROOT=/share/guozhix/wmagentattack/0824/successor_evidence_identifiability_v27/formal_v1
V17=/share/guozhix/wmagentattack/0818/paired_branch_identifiability_v17/formal_v1/paired_branch_dataset.json
V18=/share/guozhix/wmagentattack/0818/parameter_intervention_v18/formal_v1/parameter_intervention_dataset.json
V19=/share/guozhix/wmagentattack/0818/persistence_conflict_v19/formal_v2/persistence_conflict_dataset.json
UNION=/share/guozhix/wmagentattack/0821/intervention_union_v20/preflight_v1/a/dataset.json
HARD=/share/guozhix/wmagentattack/0821/hard_label_confirmation_v21/preflight_v1/a/dataset.json
SUPPORT=/share/guozhix/wmagentattack/0824/explicit_atom_support_v25/formal_v1/execution_dataset.json
cd "$WT"
test ! -e "$ROOT/COMPLETE"
test "$(git rev-parse HEAD)" = "$(cat "$ROOT/GIT_HEAD")"
sha256sum -c "$ROOT/code_and_data.prerun.sha256" > "$ROOT/prerun.verify.txt"
mkdir -p "$ROOT/build_a" "$ROOT/build_b"
PYTHONPATH=src "$PY" -m pytest -q tests/test_successor_evidence_contract.py > "$ROOT/tests.txt"
START=$(date +%s)
for BUILD in a b; do
  PYTHONPATH=src "$PY" scripts/265_build_successor_evidence_contract_v27.py \
    --protocol "$PROTOCOL" --v17 "$V17" --v18 "$V18" --v19 "$V19" \
    --union "$UNION" --hard "$HARD" --support-execution "$SUPPORT" \
    --output "$ROOT/build_${BUILD}/dataset.json" --audit "$ROOT/build_${BUILD}/audit.json" \
    > "$ROOT/build_${BUILD}/stdout.json"
done
PYTHONPATH=src "$PY" scripts/266_gate_successor_evidence_contract_v27.py \
  --protocol "$PROTOCOL" \
  --dataset-a "$ROOT/build_a/dataset.json" --audit-a "$ROOT/build_a/audit.json" \
  --dataset-b "$ROOT/build_b/dataset.json" --audit-b "$ROOT/build_b/audit.json" \
  --output "$ROOT/gate.json" > "$ROOT/gate_stdout.json"
END=$(date +%s)
printf 'start=%s\nend=%s\nelapsed_seconds=%s\n' "$START" "$END" "$((END-START))" > "$ROOT/runtime.txt"
find "$ROOT" -type f -not -name checksums.sha256 -not -name COMPLETE -print0 | sort -z | xargs -0 sha256sum > "$ROOT/checksums.sha256"
date -Is > "$ROOT/COMPLETE"
