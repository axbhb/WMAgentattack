#!/usr/bin/env bash
#SBATCH --job-name=wma-v34-paired
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --array=0-2%1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/share/guozhix/WMagentattack-paired-attack-aug26/logs/%x-%A_%a.out
#SBATCH --error=/share/guozhix/WMagentattack-paired-attack-aug26/logs/%x-%A_%a.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack-paired-attack-aug26
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
MODEL=/share/guozhix/.cache/huggingface/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b
ARCHIVE=/share/guozhix/wmagentattack/0825/paired_attack_intervention_v34/formal_v1
MANIFEST=$ARCHIVE/manifest.json
SEEDS=(503 509 521)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

cd "$ROOT"
mkdir -p "$ARCHIVE/results/seed${SEED}" "$ARCHIVE/raw/seed${SEED}" logs
export PYTHONPATH="$ROOT/src:$ROOT/external/agentdojo/src:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

"$PY" - <<'PY'
import ctypes, os, subprocess, torch
cuda = ctypes.CDLL('libcuda.so.1')
cuinit = int(cuda.cuInit(0))
ok = cuinit == 0 and torch.cuda.is_available() and torch.cuda.device_count() == 1
ok = ok and subprocess.run(['nvidia-smi'], stdout=subprocess.DEVNULL).returncode == 0
print({'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'), 'SLURM_JOB_GPUS': os.environ.get('SLURM_JOB_GPUS'), 'cuInit': cuinit, 'torch_cuda': torch.cuda.is_available(), 'device_count': torch.cuda.device_count()})
if not ok:
    raise SystemExit('CUDA allocation preflight failed')
PY

"$PY" scripts/280_run_paired_attack_intervention_v34.py \
  --manifest "$MANIFEST" \
  --model-path "$MODEL" \
  --model-label "meta-llama-3.1-70b-v34-seed${SEED}" \
  --run-seed "$SEED" \
  --protocol function_tags \
  --logdir "$ARCHIVE/raw/seed${SEED}" \
  --output "$ARCHIVE/results/seed${SEED}/result.json"

date -Iseconds > "$ARCHIVE/results/seed${SEED}/COMPLETE"
echo "WMA_V34_PAIRED_SEED_DONE seed=$SEED $(date -Is)"
