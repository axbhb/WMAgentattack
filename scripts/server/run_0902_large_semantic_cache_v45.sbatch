#!/bin/bash
#SBATCH --job-name=wm-v45-cache
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/share/guozhix/wmagentattack/0902/large_hybrid_world_model_v45/formal_v1/logs/cache-%j.out
#SBATCH --error=/share/guozhix/wmagentattack/0902/large_hybrid_world_model_v45/formal_v1/logs/cache-%j.err

set -euo pipefail

REPO=${WMAGENTATTACK_REPO:-/share/guozhix/WMagentattack-large-hybrid-sep2}
ARCHIVE=/share/guozhix/wmagentattack/0902/large_hybrid_world_model_v45/formal_v1
DATASET=/share/guozhix/wmagentattack/0817/structured_joint_outcome_v5/preflight_v1/a/dataset.json
E5=/share/guozhix/hf_cache/models--intfloat--e5-base-v2/snapshots/f52bf8ec8c7124536f0efb74aca902b2995e5bcd
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python

mkdir -p "${ARCHIVE}/logs" "${ARCHIVE}/semantic_cache"
cd "${REPO}"

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

"${PY}" - <<'PY'
import ctypes, json, os, subprocess, torch
payload = {
    "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "cuinit": int(ctypes.CDLL("libcuda.so.1").cuInit(0)),
    "torch_cuda": bool(torch.cuda.is_available()),
    "torch_device_count": int(torch.cuda.device_count()),
    "nvidia_smi": subprocess.run(["nvidia-smi"], capture_output=True).returncode,
}
print(json.dumps(payload, sort_keys=True))
if payload["cuinit"] != 0 or not payload["torch_cuda"] or payload["torch_device_count"] != 1 or payload["nvidia_smi"] != 0:
    raise SystemExit("CUDA allocation preflight failed")
PY

"${PY}" scripts/297_build_large_semantic_cache_v45.py \
  --dataset "${DATASET}" \
  --model "${E5}" \
  --output "${ARCHIVE}/semantic_cache/large_semantic_cache_v45.npz" \
  --metadata "${ARCHIVE}/semantic_cache/large_semantic_cache_v45_metadata.json" \
  --batch-size 64 \
  --max-length 256 \
  --device cuda
