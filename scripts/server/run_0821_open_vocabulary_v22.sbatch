#!/usr/bin/env bash
#SBATCH --job-name=wma-open-vocab-v22
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/share/guozhix/WMagentattack-parallel-gates-aug21/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-parallel-gates-aug21/logs/%x-%j.err

set -euo pipefail
WT=/share/guozhix/WMagentattack-parallel-gates-aug21
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
PROTOCOL=configs/0821_parallel_world_model_gates_v22_protocol.json
DATA=/share/guozhix/wmagentattack/0821/hard_label_confirmation_v21/preflight_v1/a/dataset.json
ROOT=/share/guozhix/wmagentattack/0821/parallel_world_model_gates_v22/formal_v1
OUT=$ROOT/open_vocabulary
cd "$WT"
mkdir -p logs
if [[ -e "$OUT/COMPLETE" || -e "$OUT.tmp" ]]; then echo duplicate-open-v22 >&2; exit 3; fi
mkdir -p "$OUT.tmp/results"
trap 'status=$?; echo "$status" > "$OUT.tmp/FAILED"; exit "$status"' ERR
export PYTHONPATH="$WT/src:/share/guozhix/WMagentattack/external/agentdojo/src"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
{
  echo "job_id=$SLURM_JOB_ID"
  echo "git_head=$(git rev-parse HEAD)"
  echo "started_at=$(date -Is)"
  echo "device=cpu"
} > "$OUT.tmp/runtime.txt"
"$PY" -m pytest -q \
  tests/test_compositional_effect_world_model.py \
  tests/test_paired_generation_schema.py \
  tests/test_hard_label_confirmation.py \
  tests/test_intervention_modular_world_model.py \
  > "$OUT.tmp/tests.txt"
"$PY" scripts/248_train_open_vocabulary_v22.py \
  --protocol "$PROTOCOL" --dataset "$DATA" --output-dir "$OUT.tmp/results" \
  > "$OUT.tmp/train_stdout.json"
"$PY" scripts/249_gate_open_vocabulary_v22.py \
  --protocol "$PROTOCOL" --metrics "$OUT.tmp/results/run_metrics.json" \
  --output "$OUT.tmp/results/gate.json" > "$OUT.tmp/gate_stdout.json"
echo "finished_at=$(date -Is)" >> "$OUT.tmp/runtime.txt"
touch "$OUT.tmp/COMPLETE"
mv "$OUT.tmp" "$OUT"
trap - ERR

