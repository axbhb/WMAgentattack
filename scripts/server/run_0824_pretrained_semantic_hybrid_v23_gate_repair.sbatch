#!/usr/bin/env bash
#SBATCH --job-name=wma-semantic-v23-gate
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/share/guozhix/WMagentattack-pretrained-semantic-aug24/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-pretrained-semantic-aug24/logs/%x-%j.err

set -euo pipefail
WT=/share/guozhix/WMagentattack-pretrained-semantic-aug24
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
PROTOCOL=configs/0824_pretrained_semantic_hybrid_v23_protocol.json
ROOT=/share/guozhix/wmagentattack/0824/pretrained_semantic_hybrid_v23/formal_v1
REPAIR=$ROOT/runtime_repair_v1
cd "$WT"
test ! -e "$ROOT/COMPLETE"
test ! -e "$REPAIR/COMPLETE"
test -s "$ROOT/results/run_metrics.json"
mkdir -p "$REPAIR"
cp "$WT/logs/wma-semantic-hybrid-v23-7291.out" "$REPAIR/original_7291.out"
cp "$WT/logs/wma-semantic-hybrid-v23-7291.err" "$REPAIR/original_7291.err"
git rev-parse HEAD > "$REPAIR/REPAIR_GIT_HEAD"
printf '%s\n' 'Label-blind gate-only repair: Python Boolean literals in authorization JSON.' \
  'No embedding, fit, prediction, seed, split, metric, or threshold changed.' > "$REPAIR/scope.txt"
sha256sum "$ROOT/results/run_metrics.json" > "$REPAIR/model_result.sha256"
PYTHONPATH=src "$PY" -m pytest -q tests/test_pretrained_semantic_effect_world_model.py > "$REPAIR/tests.txt"
AUDIT=$($PY -c 'import json; d=json.load(open("configs/0824_pretrained_semantic_hybrid_v23_protocol.json")); print(d["pretrained_semantic_cache"]["audit_path"])')
PYTHONPATH=src "$PY" scripts/257_gate_pretrained_semantic_hybrid_v23.py \
  --protocol "$PROTOCOL" --metrics "$ROOT/results/run_metrics.json" \
  --cache-audit "$AUDIT" --output "$ROOT/results/gate.json" > "$ROOT/gate_stdout.json"
find "$ROOT" -type f -not -name checksums.sha256 -not -name COMPLETE -print0 | sort -z | xargs -0 sha256sum > "$ROOT/checksums.sha256"
date -Is > "$REPAIR/COMPLETE"
date -Is > "$ROOT/COMPLETE"
