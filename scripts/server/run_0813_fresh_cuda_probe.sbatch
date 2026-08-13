#!/usr/bin/env bash
#SBATCH --job-name=wma-cuda-probe-0813
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/share/guozhix/wmagentattack/0813/fresh_integrated_validation/formal_v1/infrastructure_probe/probe_%j.out
#SBATCH --error=/share/guozhix/wmagentattack/0813/fresh_integrated_validation/formal_v1/infrastructure_probe/probe_%j.err

set -euo pipefail

PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
PROBE_DIR=/share/guozhix/wmagentattack/0813/fresh_integrated_validation/formal_v1/infrastructure_probe
mkdir -p "$PROBE_DIR"

export PROBE_RESULT="$PROBE_DIR/probe_${SLURM_JOB_ID}.json"

"$PY" - <<'PY'
import ctypes
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

result_path = Path(os.environ["PROBE_RESULT"])
assigned = os.environ.get("SLURM_JOB_GPUS")
visible = os.environ.get("CUDA_VISIBLE_DEVICES")
result = {
    "checked_at": datetime.now(timezone.utc).isoformat(),
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_job_gpus": assigned,
    "cuda_visible_devices": visible,
    "cuinit_status": None,
    "torch_cuda_available": False,
    "torch_cuda_device_count": 0,
    "torch_device_name": None,
    "nvidia_smi_returncode": None,
    "nvidia_smi_output": None,
    "passed": False,
    "error": None,
}

try:
    if not assigned or visible != assigned:
        raise RuntimeError(
            "Slurm GPU mapping mismatch: "
            f"SLURM_JOB_GPUS={assigned!r} CUDA_VISIBLE_DEVICES={visible!r}"
        )

    result["cuinit_status"] = ctypes.CDLL("libcuda.so.1").cuInit(0)
    if result["cuinit_status"] != 0:
        raise RuntimeError(f"CUDA driver initialization failed: cuInit={result['cuinit_status']}")

    result["torch_cuda_available"] = torch.cuda.is_available()
    result["torch_cuda_device_count"] = torch.cuda.device_count()
    if not result["torch_cuda_available"] or result["torch_cuda_device_count"] != 1:
        raise RuntimeError(
            "PyTorch CUDA probe failed: "
            f"available={result['torch_cuda_available']} "
            f"count={result['torch_cuda_device_count']}"
        )
    result["torch_device_name"] = torch.cuda.get_device_name(0)

    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    result["nvidia_smi_returncode"] = smi.returncode
    result["nvidia_smi_output"] = smi.stdout.strip()
    if smi.returncode != 0 or not result["nvidia_smi_output"]:
        raise RuntimeError(
            f"nvidia-smi failed: returncode={smi.returncode} stderr={smi.stderr.strip()!r}"
        )

    result["passed"] = True
except Exception as exc:
    result["error"] = f"{type(exc).__name__}: {exc}"
finally:
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

print(json.dumps(result, indent=2, sort_keys=True))
if not result["passed"]:
    raise SystemExit(1)
PY

echo "FRESH_CUDA_PROBE_PASS job=${SLURM_JOB_ID} $(date -Is)"
