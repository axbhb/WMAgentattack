#!/usr/bin/env bash
#SBATCH --job-name=WMA_BoundSuccV28
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/share/guozhix/WMagentattack-bound-successor-aug24/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-bound-successor-aug24/logs/%x-%j.err

set -euo pipefail

WT=/share/guozhix/WMagentattack-bound-successor-aug24
ARCHIVE=/share/guozhix/wmagentattack/0824/bound_successor_records_v28/formal_v1
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
PROTOCOL=$WT/configs/0824_bound_successor_records_v28_protocol.json
HARD=/share/guozhix/wmagentattack/0821/hard_label_confirmation_v21/preflight_v1/a/dataset.json
STRUCTURED=/share/guozhix/wmagentattack/0824/successor_evidence_identifiability_v27/formal_v1/build_a/dataset.json
V26=/share/guozhix/wmagentattack/0824/support_conditioned_compositional_v26/formal_v1/results/run_metrics.json
V27_GATE=/share/guozhix/wmagentattack/0824/successor_evidence_identifiability_v27/formal_v1/gate.json

mkdir -p "$ARCHIVE/logs" "$ARCHIVE/results" "$ARCHIVE/source_snapshot"
cd "$WT"
test ! -e "$ARCHIVE/COMPLETE"
test "$(git rev-parse HEAD)" = "$(cat "$ARCHIVE/GIT_HEAD")"
sha256sum -c "$ARCHIVE/code_and_data.prerun.sha256" > "$ARCHIVE/prerun.verify.txt"

start=$(date +%s)
PYTHONPATH=src "$PY" -m pytest -q \
  tests/test_bound_successor_world_model.py \
  tests/test_successor_evidence_contract.py | tee "$ARCHIVE/tests.txt"
PYTHONPATH=src "$PY" scripts/267_train_bound_successor_records_v28.py \
  --protocol "$PROTOCOL" \
  --hard-dataset "$HARD" \
  --structured-dataset "$STRUCTURED" \
  --output-dir "$ARCHIVE/results" > "$ARCHIVE/train_stdout.json"
PYTHONPATH=src "$PY" scripts/268_gate_bound_successor_records_v28.py \
  --protocol "$PROTOCOL" \
  --metrics "$ARCHIVE/results/run_metrics.json" \
  --v26-metrics "$V26" \
  --v27-gate "$V27_GATE" \
  --output "$ARCHIVE/results/gate.json" > "$ARCHIVE/gate_stdout.json"
end=$(date +%s)
printf 'start=%s\nend=%s\nelapsed_seconds=%s\n' "$start" "$end" "$((end-start))" > "$ARCHIVE/runtime.txt"
find "$ARCHIVE" -type f ! -name checksums.sha256 -print0 | sort -z | xargs -0 sha256sum > "$ARCHIVE/checksums.sha256"
printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" > "$ARCHIVE/COMPLETE"
