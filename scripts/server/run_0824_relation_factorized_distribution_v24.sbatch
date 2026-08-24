#!/usr/bin/env bash
#SBATCH --job-name=wma-relation-v24
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/share/guozhix/WMagentattack-relational-support-aug24/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-relational-support-aug24/logs/%x-%j.err

set -euo pipefail
WT=/share/guozhix/WMagentattack-relational-support-aug24
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
PROTOCOL=configs/0824_relation_factorized_distribution_v24_protocol.json
ROOT=/share/guozhix/wmagentattack/0824/relation_factorized_distribution_v24/formal_v1
DATA=/share/guozhix/wmagentattack/0821/hard_label_confirmation_v21/preflight_v1/a/dataset.json
cd "$WT"
test ! -e "$ROOT/COMPLETE"
test "$(git rev-parse HEAD)" = "$(cat "$ROOT/GIT_HEAD")"
sha256sum -c "$ROOT/code_and_data.prerun.sha256" > "$ROOT/prerun.verify.txt"
CACHE=$($PY -c 'import json; d=json.load(open("configs/0824_relation_factorized_distribution_v24_protocol.json")); print(d["semantic_cache"]["cache_path"])')
AUDIT=$($PY -c 'import json; d=json.load(open("configs/0824_relation_factorized_distribution_v24_protocol.json")); print(d["semantic_cache"]["audit_path"])')
mkdir -p "$ROOT/results"
PYTHONPATH=src "$PY" -m pytest -q tests/test_relation_factorized_semantic_world_model.py > "$ROOT/tests.txt"
START=$(date +%s)
PYTHONPATH=src "$PY" scripts/259_train_relation_factorized_hybrid_v24.py --protocol "$PROTOCOL" --dataset "$DATA" --semantic-cache "$CACHE" --output-dir "$ROOT/results" > "$ROOT/train_stdout.json"
PYTHONPATH=src "$PY" scripts/260_gate_relation_factorized_hybrid_v24.py --protocol "$PROTOCOL" --metrics "$ROOT/results/run_metrics.json" --cache-audit "$AUDIT" --output "$ROOT/results/gate.json" > "$ROOT/gate_stdout.json"
END=$(date +%s)
printf 'start=%s\nend=%s\nelapsed_seconds=%s\n' "$START" "$END" "$((END-START))" > "$ROOT/runtime.txt"
find "$ROOT" -type f -not -name checksums.sha256 -not -name COMPLETE -print0 | sort -z | xargs -0 sha256sum > "$ROOT/checksums.sha256"
date -Is > "$ROOT/COMPLETE"
