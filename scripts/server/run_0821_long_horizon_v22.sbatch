#!/usr/bin/env bash
#SBATCH --job-name=wma-long-horizon-v22
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/share/guozhix/WMagentattack-parallel-gates-aug21/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-parallel-gates-aug21/logs/%x-%j.err

set -euo pipefail
WT=/share/guozhix/WMagentattack-parallel-gates-aug21
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
PROTOCOL=configs/0821_parallel_world_model_gates_v22_protocol.json
ADJACENT=/share/guozhix/wmagentattack/0813/adjacent_transition/preflight_v1/a/dataset.json
DATA=/share/guozhix/wmagentattack/0817/structured_joint_outcome_v5/preflight_v1/a/dataset.json
AUDIT=/share/guozhix/wmagentattack/0817/structured_joint_outcome_v5/preflight_v1/a/audit.json
ROOT=/share/guozhix/wmagentattack/0821/parallel_world_model_gates_v22/formal_v2
OUT=$ROOT/long_horizon
cd "$WT"
mkdir -p logs
test "$(git rev-parse HEAD)" = "$(cat "$ROOT/GIT_HEAD")"
sha256sum -c "$ROOT/code_and_data.prerun.sha256" > "$OUT.prerun.verify.txt"
if [[ -e "$OUT/COMPLETE" || -e "$OUT.tmp" ]]; then echo duplicate-long-v22 >&2; exit 3; fi
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
"$PY" scripts/250_audit_long_horizon_v22.py \
  --protocol "$PROTOCOL" --dataset "$ADJACENT" --output "$OUT.tmp/results/data_gate.json" \
  > "$OUT.tmp/data_gate_stdout.json"
"$PY" - "$OUT.tmp/results/data_gate.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d['decision']=='GO_LONG_HORIZON_MODEL_GATE_V22', d
PY
"$PY" scripts/251_train_long_horizon_controls_v22.py \
  --protocol "$PROTOCOL" --dataset "$DATA" --audit "$AUDIT" \
  --output-dir "$OUT.tmp/results" > "$OUT.tmp/train_stdout.json"
echo "finished_at=$(date -Is)" >> "$OUT.tmp/runtime.txt"
sha256sum -c "$ROOT/code_and_data.prerun.sha256" > "$OUT.tmp/postrun.verify.txt"
touch "$OUT.tmp/COMPLETE"
mv "$OUT.tmp" "$OUT"
trap - ERR
