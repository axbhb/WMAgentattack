#!/usr/bin/env bash
#SBATCH --job-name=wma-clean-ms-8b
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
MODEL_LABEL_BASE="${MODEL_LABEL_BASE:-meta-llama-3.1-8b-instruct-clean-ms}"
SEEDS="${SEEDS:-7 13 21}"
SUITES="${SUITES:-workspace slack travel banking}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/runs/agentdojo_clean_multiseed_llama31_8b}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"

mkdir -p "$PROJECT_ROOT/logs" "$LOG_ROOT"
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

for seed in $SEEDS; do
  python scripts/10_run_agentdojo_hf_full.py \
    --model-path "$MODEL_PATH" \
    --model-label "${MODEL_LABEL_BASE}-seed${seed}" \
    --quantization 4bit \
    --protocol function_tags \
    --prompt-profile base \
    --max-input-tokens 8192 \
    --max-new-tokens 256 \
    --max-tool-output-chars 12000 \
    --agentdojo-local-alias \
    --clean-only \
    --force-rerun \
    --seed "$seed" \
    --do-sample \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --logdir "$LOG_ROOT/seed${seed}" \
    "${suite_args[@]}"
done

python scripts/20_summarize_clean_solvability.py \
  --clean-root "$LOG_ROOT" \
  --output "$PROJECT_ROOT/artifacts/clean_multiseed_llama31_8b_solvability.json"

