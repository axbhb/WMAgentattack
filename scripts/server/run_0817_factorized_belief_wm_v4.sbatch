#!/bin/bash
#SBATCH --job-name=wm-fns-bwm-v4-cpu
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/0817_factorized_belief_wm_v4_%j.out
#SBATCH --error=logs/0817_factorized_belief_wm_v4_%j.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack-v4-pilot-aug17
ARCHIVE=/share/guozhix/wmagentattack/0817/factorized_belief_wm_v4/pilot_v1
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
PROTOCOL=$ROOT/configs/0817_factorized_belief_wm_v4_pilot.json
DATA=/share/guozhix/wmagentattack/0813/adjacent_transition/preflight_v1/a/dataset.json
AUDIT=/share/guozhix/wmagentattack/0813/adjacent_transition/preflight_v1/a/audit.json

cd "$ROOT"
export PYTHONPATH="$ROOT/src"
mkdir -p logs
if [[ -e "$ARCHIVE/COMPLETE" ]]; then
  echo "Refusing duplicate completed experiment: $ARCHIVE" >&2
  exit 3
fi
rm -rf "$ARCHIVE.tmp"
mkdir -p "$ARCHIVE.tmp/results" "$ARCHIVE.tmp/source_snapshot"

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "runtime_device=cpu"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "slurm_job_gpus=${SLURM_JOB_GPUS:-}"
  "$PY" -c 'import torch; print(torch.__version__, torch.get_num_threads())'
} > "$ARCHIVE.tmp/runtime_preflight.txt"

cp "$PROTOCOL" "$ARCHIVE.tmp/protocol.frozen.json"
cp src/wmagentattack/factorized_belief_world_model.py "$ARCHIVE.tmp/source_snapshot/"
cp scripts/197_train_factorized_belief_wm_v4.py "$ARCHIVE.tmp/source_snapshot/"
cp scripts/198_summarize_factorized_belief_wm_v4.py "$ARCHIVE.tmp/source_snapshot/"
cp scripts/199_audit_factorized_belief_wm_v4.py "$ARCHIVE.tmp/source_snapshot/"
cp scripts/server/run_0817_factorized_belief_wm_v4.sbatch "$ARCHIVE.tmp/source_snapshot/"
sha256sum \
  "$PROTOCOL" "$DATA" "$AUDIT" \
  src/wmagentattack/factorized_belief_world_model.py \
  scripts/197_train_factorized_belief_wm_v4.py \
  scripts/198_summarize_factorized_belief_wm_v4.py \
  scripts/199_audit_factorized_belief_wm_v4.py \
  scripts/server/run_0817_factorized_belief_wm_v4.sbatch \
  > "$ARCHIVE.tmp/frozen_inputs.sha256"

"$PY" -m pytest \
  tests/test_factorized_belief_world_model.py \
  tests/test_adjacent_transition.py \
  -q > "$ARCHIVE.tmp/preflight_tests.txt"

"$PY" scripts/199_audit_factorized_belief_wm_v4.py \
  --protocol "$PROTOCOL" \
  --dataset "$DATA" \
  --audit "$AUDIT" \
  --output "$ARCHIVE.tmp/representation_audit.json" \
  > "$ARCHIVE.tmp/representation_audit_stdout.json"

"$PY" scripts/197_train_factorized_belief_wm_v4.py \
  --protocol "$PROTOCOL" \
  --dataset "$DATA" \
  --audit "$AUDIT" \
  --output-dir "$ARCHIVE.tmp/results" \
  --device cpu \
  > "$ARCHIVE.tmp/training_stdout.json"

"$PY" scripts/198_summarize_factorized_belief_wm_v4.py \
  --protocol "$PROTOCOL" \
  --dataset "$DATA" \
  --audit "$AUDIT" \
  --predictions "$ARCHIVE.tmp/results/predictions.jsonl" \
  --run-metrics "$ARCHIVE.tmp/results/run_metrics.json" \
  --output "$ARCHIVE.tmp/results/summary.json" \
  --markdown "$ARCHIVE.tmp/results/results.md" \
  > "$ARCHIVE.tmp/summary_stdout.json"

sha256sum -c "$ARCHIVE.tmp/frozen_inputs.sha256" \
  > "$ARCHIVE.tmp/frozen_inputs.postrun.verify.txt"
echo "runtime_failures=0" >> "$ARCHIVE.tmp/runtime_preflight.txt"
touch "$ARCHIVE.tmp/COMPLETE"
(
  cd "$ARCHIVE.tmp"
  find . -type f ! -name checksums.sha256 -print0 \
    | sort -z | xargs -0 sha256sum > checksums.sha256
)
mv "$ARCHIVE.tmp" "$ARCHIVE"
echo "FNS_BWM_V4_PILOT_DONE"
