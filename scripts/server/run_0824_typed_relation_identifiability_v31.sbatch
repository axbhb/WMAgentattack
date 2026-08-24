#!/usr/bin/env bash
#SBATCH --job-name=wma-typed-rel-v31
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=/share/guozhix/WMagentattack-typed-relations-aug24/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-typed-relations-aug24/logs/%x-%j.err

set -euo pipefail
WT=/share/guozhix/WMagentattack-typed-relations-aug24
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
PROTOCOL=configs/0824_typed_relation_identifiability_v31_protocol.json
ROOT=/share/guozhix/wmagentattack/0824/typed_relation_identifiability_v31/formal_v1
HARD=/share/guozhix/wmagentattack/0821/hard_label_confirmation_v21/preflight_v1/a/dataset.json
RELATIONAL=/share/guozhix/wmagentattack/0824/relational_successor_data_v29/formal_v1/build_a/dataset.json
cd "$WT"
test ! -e "$ROOT/COMPLETE"
test "$(git rev-parse HEAD)" = "$(cat "$ROOT/GIT_HEAD")"
sha256sum -c "$ROOT/code_and_data.prerun.sha256" > "$ROOT/prerun.verify.txt"
PYTHONPATH=src "$PY" -m pytest -q tests/test_typed_relation_contract.py tests/test_relational_successor_contract.py > "$ROOT/tests.txt"
START=$(date +%s)
for replica in a b; do
  PYTHONPATH=src "$PY" scripts/273_build_typed_relation_identifiability_v31.py \
    --protocol "$PROTOCOL" --hard-dataset "$HARD" --relational-dataset "$RELATIONAL" \
    --output-dir "$ROOT/build_$replica" > "$ROOT/build_$replica.stdout.json"
done
PYTHONPATH=src "$PY" scripts/274_gate_typed_relation_identifiability_v31.py \
  --protocol "$PROTOCOL" --build-a "$ROOT/build_a" --build-b "$ROOT/build_b" \
  --output "$ROOT/gate.json" > "$ROOT/gate_stdout.json"
END=$(date +%s)
printf 'start=%s\nend=%s\nelapsed_seconds=%s\n' "$START" "$END" "$((END-START))" > "$ROOT/runtime.txt"
find "$ROOT" -type f -not -name checksums.sha256 -not -name COMPLETE -print0 | sort -z | xargs -0 sha256sum > "$ROOT/checksums.sha256"
date -Is > "$ROOT/COMPLETE"
