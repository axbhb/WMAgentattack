#!/usr/bin/env bash
#SBATCH --job-name=wmagent_llama31_8b
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/guozhix/WMagentattack}"
CONDA_BIN="${CONDA_BIN:-/share/guozhix/software/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-wmagentattack}"
MODEL_PATH="${MODEL_PATH:-/share/guozhix/.cache/huggingface/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659}"
MODEL_LABEL="${MODEL_LABEL:-meta-llama-3.1-8b-instruct}"
ATTACKS="${ATTACKS:-important_instructions_no_model_name}"
SUITES="${SUITES:-workspace slack travel banking}"
LOGDIR="${LOGDIR:-$PROJECT_ROOT/runs/agentdojo_full_hf}"

mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"

eval "$("$CONDA_BIN" shell.bash hook)"
conda activate "$ENV_NAME"

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/external/agentdojo/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/share/guozhix/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

suite_args=()
for suite in $SUITES; do
  suite_args+=(--suite "$suite")
done

attack_args=()
for attack in $ATTACKS; do
  attack_args+=(--attack "$attack")
done

python scripts/10_run_agentdojo_hf_full.py \
  --model-path "$MODEL_PATH" \
  --model-label "$MODEL_LABEL" \
  --quantization 4bit \
  --protocol function_tags \
  --prompt-profile base \
  --max-input-tokens 8192 \
  --max-new-tokens 256 \
  --max-tool-output-chars 12000 \
  --agentdojo-local-alias \
  --logdir "$LOGDIR" \
  "${suite_args[@]}" \
  "${attack_args[@]}"

