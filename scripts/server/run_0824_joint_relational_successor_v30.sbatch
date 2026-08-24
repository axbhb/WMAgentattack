#!/usr/bin/env bash
#SBATCH --job-name=wma-joint-relational-v30
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:40:00
#SBATCH --output=/share/guozhix/WMagentattack-joint-relational-aug24/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-joint-relational-aug24/logs/%x-%j.err

set -euo pipefail
WT=/share/guozhix/WMagentattack-joint-relational-aug24
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
PROTOCOL=configs/0824_joint_relational_successor_v30_protocol.json
ROOT=/share/guozhix/wmagentattack/0824/joint_relational_successor_v30/formal_v1
HARD=/share/guozhix/wmagentattack/0821/hard_label_confirmation_v21/preflight_v1/a/dataset.json
STRUCTURED=/share/guozhix/wmagentattack/0824/successor_evidence_identifiability_v27/formal_v1/build_a/dataset.json
RELATIONAL=/share/guozhix/wmagentattack/0824/relational_successor_data_v29/formal_v1/build_a/dataset.json
V28=/share/guozhix/wmagentattack/0824/bound_successor_records_v28/formal_v1/results/run_metrics.json
V29_GATE=/share/guozhix/wmagentattack/0824/relational_successor_data_v29/formal_v1/gate.json
cd "$WT"
test ! -e "$ROOT/COMPLETE"
test "$(git rev-parse HEAD)" = "$(cat "$ROOT/GIT_HEAD")"
sha256sum -c "$ROOT/code_and_data.prerun.sha256" > "$ROOT/prerun.verify.txt"
mkdir -p "$ROOT/results"
PYTHONPATH=src "$PY" -m pytest -q tests/test_joint_relational_world_model.py tests/test_relational_successor_contract.py > "$ROOT/tests.txt"
START=$(date +%s)
PYTHONPATH=src "$PY" scripts/271_train_joint_relational_successor_v30.py \
  --protocol "$PROTOCOL" --hard-dataset "$HARD" --structured-dataset "$STRUCTURED" \
  --relational-dataset "$RELATIONAL" --output-dir "$ROOT/results" > "$ROOT/train_stdout.json"
PYTHONPATH=src "$PY" scripts/272_gate_joint_relational_successor_v30.py \
  --protocol "$PROTOCOL" --metrics "$ROOT/results/run_metrics.json" \
  --v28-metrics "$V28" --v29-gate "$V29_GATE" --output "$ROOT/gate.json" > "$ROOT/gate_stdout.json"
END=$(date +%s)
printf 'start=%s\nend=%s\nelapsed_seconds=%s\n' "$START" "$END" "$((END-START))" > "$ROOT/runtime.txt"
find "$ROOT" -type f -not -name checksums.sha256 -not -name COMPLETE -print0 | sort -z | xargs -0 sha256sum > "$ROOT/checksums.sha256"
date -Is > "$ROOT/COMPLETE"
