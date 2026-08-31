#!/usr/bin/env bash
#SBATCH --job-name=wma-v40-confirm
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/share/guozhix/WMagentattack-protocol-recovery-aug31/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-protocol-recovery-aug31/logs/%x-%j.err
set -euo pipefail
ROOT=/share/guozhix/WMagentattack-protocol-recovery-aug31
ARCHIVE=/share/guozhix/wmagentattack/0831/protocol_recovery_v40/formal_v1
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/external/agentdojo/src"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$PY" -m pytest -q tests/test_protocol_recovery.py tests/test_protocol_recovery_eval.py tests/test_protocol_recovery_adapter.py tests/test_qwen_adapter.py tests/test_clean_pairing.py
"$PY" scripts/289_run_protocol_recovery_v39_v40.py --protocol "$ARCHIVE/preregistered_protocol.json" --stage v40
date -Iseconds > "$ARCHIVE/COMPLETE"
