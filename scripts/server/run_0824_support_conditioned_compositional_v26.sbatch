#!/usr/bin/env bash
#SBATCH --job-name=wma-support-compositional-v26
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/share/guozhix/WMagentattack-support-conditioned-aug24/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-support-conditioned-aug24/logs/%x-%j.err

set -euo pipefail
WT=/share/guozhix/WMagentattack-support-conditioned-aug24
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
PROTOCOL=configs/0824_support_conditioned_compositional_v26_protocol.json
ROOT=/share/guozhix/wmagentattack/0824/support_conditioned_compositional_v26/formal_v1
DATA=/share/guozhix/wmagentattack/0821/hard_label_confirmation_v21/preflight_v1/a/dataset.json
SUPPORT=/share/guozhix/wmagentattack/0824/explicit_atom_support_v25/formal_v1/atom_support_dataset.json
V23=/share/guozhix/wmagentattack/0824/pretrained_semantic_hybrid_v23/formal_v1/results/run_metrics.json
cd "$WT"
test ! -e "$ROOT/COMPLETE"
test "$(git rev-parse HEAD)" = "$(cat "$ROOT/GIT_HEAD")"
sha256sum -c "$ROOT/code_and_data.prerun.sha256" > "$ROOT/prerun.verify.txt"
mkdir -p "$ROOT/results"
PYTHONPATH=src "$PY" -m pytest -q tests/test_support_conditioned_effect_world_model.py > "$ROOT/tests.txt"
START=$(date +%s)
PYTHONPATH=src "$PY" scripts/263_train_support_conditioned_compositional_v26.py \
  --protocol "$PROTOCOL" --dataset "$DATA" --support-dataset "$SUPPORT" \
  --output-dir "$ROOT/results" > "$ROOT/train_stdout.json"
PYTHONPATH=src "$PY" scripts/264_gate_support_conditioned_compositional_v26.py \
  --protocol "$PROTOCOL" --metrics "$ROOT/results/run_metrics.json" \
  --v23-metrics "$V23" --output "$ROOT/results/gate.json" > "$ROOT/gate_stdout.json"
END=$(date +%s)
printf 'start=%s\nend=%s\nelapsed_seconds=%s\n' "$START" "$END" "$((END-START))" > "$ROOT/runtime.txt"
find "$ROOT" -type f -not -name checksums.sha256 -not -name COMPLETE -print0 | sort -z | xargs -0 sha256sum > "$ROOT/checksums.sha256"
date -Is > "$ROOT/COMPLETE"
