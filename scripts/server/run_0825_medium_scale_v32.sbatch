#!/usr/bin/env bash
#SBATCH --job-name=wma-med-v32
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/wma-med-v32-%j.out
#SBATCH --error=logs/wma-med-v32-%j.err

set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
WT=/share/guozhix/WMagentattack-medium-scale-aug25
ARCH=/share/guozhix/wmagentattack/0825/medium_scale_v32/formal_v1
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
PROTOCOL="$WT/configs/0825_medium_scale_v32_protocol.json"
ACTION_DATA=/share/guozhix/wmagentattack/0817/structured_joint_outcome_v5/preflight_v1/a/dataset.json
ACTION_AUDIT=/share/guozhix/wmagentattack/0817/structured_joint_outcome_v5/preflight_v1/a/audit.json
EFFECT_DATA=/share/guozhix/wmagentattack/0821/hard_label_confirmation_v21/preflight_v1/a/dataset.json
SMALL_ACTION=/share/guozhix/wmagentattack/0821/parallel_world_model_gates_v22/formal_v2/long_horizon/results/predictions.jsonl
SMALL_EFFECT=/share/guozhix/wmagentattack/0821/hard_label_confirmation_v21/formal_v1/results/run_metrics.json

cd "$WT"
mkdir -p logs "$ARCH/results"
date +%s > "$ARCH/start.epoch"
echo "$SLURM_JOB_ID" > "$ARCH/SLURM_JOB_ID"
git rev-parse HEAD > "$ARCH/GIT_HEAD"
sha256sum -c "$ARCH/code_and_data.prerun.sha256"

"$PY" - <<'PY' | tee "$ARCH/cuda_preflight.json"
import ctypes, json, os, subprocess, torch
cuda = ctypes.CDLL('libcuda.so.1')
cuinit = int(cuda.cuInit(0))
payload = {
    'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
    'SLURM_JOB_GPUS': os.environ.get('SLURM_JOB_GPUS'),
    'cuInit': cuinit,
    'torch_cuda': bool(torch.cuda.is_available()),
    'torch_device_count': int(torch.cuda.device_count()),
    'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'nvidia_smi': subprocess.run(['nvidia-smi'], capture_output=True, text=True).returncode,
}
print(json.dumps(payload, sort_keys=True))
if cuinit != 0 or not payload['torch_cuda'] or payload['torch_device_count'] != 1 or payload['nvidia_smi'] != 0:
    raise SystemExit('CUDA preflight failed')
PY

PYTHONPATH=src "$PY" -m pytest -q \
  tests/test_medium_scale_diagnostic.py \
  tests/test_structured_residual_dynamics.py \
  tests/test_hard_label_confirmation.py | tee "$ARCH/tests.txt"

"$PY" scripts/275_train_medium_scale_v32.py \
  --protocol "$PROTOCOL" \
  --action-dataset "$ACTION_DATA" \
  --action-audit "$ACTION_AUDIT" \
  --effect-dataset "$EFFECT_DATA" \
  --output-dir "$ARCH/results" | tee "$ARCH/train_stdout.json"

"$PY" scripts/276_gate_medium_scale_v32.py \
  --protocol "$PROTOCOL" \
  --metrics "$ARCH/results/run_metrics.json" \
  --action-predictions "$ARCH/results/action_predictions.jsonl" \
  --small-action-predictions "$SMALL_ACTION" \
  --small-effect-metrics "$SMALL_EFFECT" \
  --output "$ARCH/gate.json" | tee "$ARCH/gate_stdout.json"

cp "$PROTOCOL" "$ARCH/"
cp reports/0825_medium_scale_v32_preregistration.md "$ARCH/"
date +%s > "$ARCH/end.epoch"
find "$ARCH" -type f ! -name checksums.sha256 -print0 | sort -z | xargs -0 sha256sum > "$ARCH/checksums.sha256"
date -Iseconds > "$ARCH/COMPLETE"
