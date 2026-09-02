#!/usr/bin/env bash
#SBATCH --job-name=wma-mentor-v44
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/wma-mentor-v44-%j.out
#SBATCH --error=logs/wma-mentor-v44-%j.err

set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
WT=/share/guozhix/WMagentattack-mentor-scale-sep2
ARCH=/share/guozhix/wmagentattack/0902/mentor_scale_best_wm_v44/formal_v1
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
PROTOCOL="$WT/configs/0902_mentor_scale_best_wm_v44_protocol.json"
ACTION_DATA=/share/guozhix/wmagentattack/0817/structured_joint_outcome_v5/preflight_v1/a/dataset.json
ACTION_AUDIT=/share/guozhix/wmagentattack/0817/structured_joint_outcome_v5/preflight_v1/a/audit.json
EFFECT_DATA=/share/guozhix/wmagentattack/0821/hard_label_confirmation_v21/preflight_v1/a/dataset.json

cd "$WT"
mkdir -p logs "$ARCH/results"
date -Iseconds > "$ARCH/STARTED"
echo "$SLURM_JOB_ID" > "$ARCH/SLURM_JOB_ID"
git rev-parse HEAD > "$ARCH/GIT_HEAD"

"$PY" - <<'PY' | tee "$ARCH/cuda_preflight.json"
import ctypes, json, os, subprocess, torch
cuda = ctypes.CDLL('libcuda.so.1')
payload = {
    'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
    'SLURM_JOB_GPUS': os.environ.get('SLURM_JOB_GPUS'),
    'cuInit': int(cuda.cuInit(0)),
    'torch_cuda': bool(torch.cuda.is_available()),
    'torch_device_count': int(torch.cuda.device_count()),
    'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'nvidia_smi': subprocess.run(['nvidia-smi'], capture_output=True, text=True).returncode,
}
print(json.dumps(payload, sort_keys=True))
if payload['cuInit'] != 0 or not payload['torch_cuda'] or payload['torch_device_count'] != 1 or payload['nvidia_smi'] != 0:
    raise SystemExit('CUDA preflight failed')
PY

PYTHONPATH=src "$PY" -m pytest -q \
  tests/test_mentor_scale_v44.py \
  tests/test_structured_joint_model_context.py \
  tests/test_structured_residual_dynamics.py \
  tests/test_hard_label_confirmation.py | tee "$ARCH/tests.txt"

"$PY" scripts/294_train_mentor_scale_best_wm_v44.py \
  --protocol "$PROTOCOL" \
  --action-dataset "$ACTION_DATA" \
  --action-audit "$ACTION_AUDIT" \
  --effect-dataset "$EFFECT_DATA" \
  --output-dir "$ARCH/results" | tee "$ARCH/train_stdout.json"

"$PY" scripts/295_gate_mentor_scale_best_wm_v44.py \
  --protocol "$PROTOCOL" \
  --metrics "$ARCH/results/run_metrics.json" \
  --joint-predictions "$ARCH/results/joint_predictions.jsonl" \
  --action-predictions "$ARCH/results/action_predictions.jsonl" \
  --output "$ARCH/gate.json" | tee "$ARCH/gate_stdout.json"

cp "$PROTOCOL" "$ARCH/"
cp reports/0902_mentor_scale_best_wm_v44_preregistration.md "$ARCH/"
date -Iseconds > "$ARCH/COMPLETE"
