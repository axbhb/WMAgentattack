#!/bin/bash
#SBATCH --job-name=wmagent-selected-8b
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/share/guozhix/WMagentattack/logs/wmagent_selected_8b-%j.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/wmagent_selected_8b-%j.err

set -euo pipefail

PROJECT_ROOT=/share/guozhix/WMagentattack
CONDA=/share/guozhix/software/miniconda3/bin/conda
MODEL_PATH=/share/guozhix/.cache/huggingface/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
RUN_ROOT=${PROJECT_ROOT}/runs/agentdojo_full_hf
TOP_K=${TOP_K:-12}
SCORING_MODE=${SCORING_MODE:-clean_prefix}
OUTPUT_PREFIX=agentdojo_full_llama31_8b_selected_${SCORING_MODE}_top${TOP_K}
SELECTION_NAMES=${SELECTION_NAMES:-"world_model_top random low_score"}
SELECTION_NAME_ARGS=()
for selection_name in ${SELECTION_NAMES}; do
  SELECTION_NAME_ARGS+=(--selection-name "${selection_name}")
done

cd "${PROJECT_ROOT}"
mkdir -p logs artifacts runs/agentdojo_selected_replay

export PYTHONPATH=${PROJECT_ROOT}/src:${PROJECT_ROOT}/external/agentdojo/src
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

"${CONDA}" run -n wmagentattack python scripts/11_select_world_model_agentdojo_pairs.py \
  --run-root "${RUN_ROOT}" \
  --model artifacts/agentdojo_full_llama31_8b_world_model.joblib \
  --output artifacts/${OUTPUT_PREFIX}_pairs.json \
  --allowed-trajectories data/agentdojo_full_llama31_8b/splits/test_trajectories.jsonl \
  --scoring-mode "${SCORING_MODE}" \
  --top-k "${TOP_K}" \
  --max-per-user-task 2 \
  --exclude-world-model-from-baselines

"${CONDA}" run -n wmagentattack python scripts/12_run_agentdojo_hf_selected.py \
  --selection artifacts/${OUTPUT_PREFIX}_pairs.json \
  "${SELECTION_NAME_ARGS[@]}" \
  --model-path "${MODEL_PATH}" \
  --model-label meta-llama-3.1-8b-instruct \
  --benchmark-version v1.2.2 \
  --quantization 4bit \
  --prompt-profile base \
  --protocol function_tags \
  --max-input-tokens 8192 \
  --max-new-tokens 256 \
  --max-tool-output-chars 12000 \
  --agentdojo-local-alias \
  --logdir runs/agentdojo_selected_replay/${SCORING_MODE}_top${TOP_K} \
  --output artifacts/${OUTPUT_PREFIX}_replay.json \
  --force-rerun
