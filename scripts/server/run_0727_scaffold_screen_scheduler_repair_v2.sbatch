#!/usr/bin/env bash
#SBATCH --job-name=wma-scaffold-v2
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --array=0-11%4
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%A_%a.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%A_%a.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
MODEL=/share/guozhix/.cache/huggingface/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b
ARCHIVE=/share/guozhix/wmagentattack/0727/scaffold_screen/scheduler_repair_v2
MANIFEST=$ARCHIVE/manifest.json
CANDIDATES=(base_sampled base_greedy constraint_checklist_sampled constraint_checklist_greedy)
PROMPTS=(base base constraint_checklist constraint_checklist)
SAMPLED=(1 0 1 0)
SEEDS=(151 157 163)
CANDIDATE_GROUP=$((SLURM_ARRAY_TASK_ID / 6))
WITHIN_GROUP=$((SLURM_ARRAY_TASK_ID % 6))
SEED_INDEX=$((WITHIN_GROUP / 2))
CHUNK=$((WITHIN_GROUP % 2))
RUN_SEED=${SEEDS[$SEED_INDEX]}
FIRST_CANDIDATE_INDEX=$((CANDIDATE_GROUP * 2))

cd "$ROOT"
mkdir -p logs
test -s "$MANIFEST"
test -s "$ARCHIVE/frozen_repo_SHA256SUMS"
test -s "$MODEL/config.json"
sha256sum --quiet -c "$ARCHIVE/frozen_repo_SHA256SUMS"
export PYTHONPATH="$ROOT/src:$ROOT/external/agentdojo/src:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

FINAL_STATUS=0
for OFFSET in 0 1; do
  CANDIDATE_INDEX=$((FIRST_CANDIDATE_INDEX + OFFSET))
  CANDIDATE=${CANDIDATES[$CANDIDATE_INDEX]}
  PROMPT=${PROMPTS[$CANDIDATE_INDEX]}
  OUTPUT=$ARCHIVE/$CANDIDATE/seed${RUN_SEED}/chunk${CHUNK}.json
  LOGDIR=$ARCHIVE/$CANDIDATE/seed${RUN_SEED}/runs/chunk${CHUNK}
  mkdir -p "$(dirname "$OUTPUT")" "$LOGDIR"
  SAMPLE_ARGS=()
  if [[ "${SAMPLED[$CANDIDATE_INDEX]}" == "1" ]]; then
    SAMPLE_ARGS=(--do-sample --temperature 0.7 --top-p 0.95)
  fi
  set +e
  "$PY" scripts/99_run_selected_clean_agentdojo.py \
    --manifest "$MANIFEST" \
    --model-path "$MODEL" \
    --model-label "meta-llama-3.1-70b-${CANDIDATE}-seed${RUN_SEED}" \
    --benchmark-version v1.2.2 \
    --quantization 4bit \
    --prompt-profile "$PROMPT" \
    --protocol function_tags_repair_retry \
    --max-input-tokens 8192 \
    --max-new-tokens 256 \
    --max-tool-output-chars 12000 \
    --agentdojo-local-alias \
    --run-seed "$RUN_SEED" \
    --chunk-index "$CHUNK" \
    --num-chunks 2 \
    "${SAMPLE_ARGS[@]}" \
    --continue-on-error \
    --logdir "$LOGDIR" \
    --output "$OUTPUT"
  STATUS=$?
  set -e
  if [[ "$STATUS" -ne 0 ]]; then
    FINAL_STATUS=$STATUS
    echo "SCAFFOLD_SCREEN_CANDIDATE_FAILED candidate=$CANDIDATE seed=$RUN_SEED chunk=$CHUNK status=$STATUS" >&2
  else
    echo "SCAFFOLD_SCREEN_CANDIDATE_DONE candidate=$CANDIDATE seed=$RUN_SEED chunk=$CHUNK $(date -Is)"
  fi
done

exit "$FINAL_STATUS"
