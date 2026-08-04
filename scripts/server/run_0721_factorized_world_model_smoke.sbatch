#!/usr/bin/env bash
#SBATCH --job-name=wma-factorized-smoke
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%j.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
SOURCE=/share/guozhix/wmagentattack/0714/agentdojo_v2/final/raw_dataset
ARCHIVE=/share/guozhix/wmagentattack/0721/factorized_refactor/fixed_budget_v1
DATA=$ARCHIVE/data/v2_joint_labels

cd "$ROOT"
test ! -e "$ARCHIVE/COMPLETE"
mkdir -p "$ARCHIVE" "$ARCHIVE/io_hmm" "$ARCHIVE/event" logs
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

{
  echo "job_id=$SLURM_JOB_ID"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "git_head=$(git rev-parse HEAD)"
  "$PY" --version
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
} > "$ARCHIVE/runtime.txt"

sha256sum \
  src/wmagentattack/decision_state.py \
  src/wmagentattack/exact_simulator.py \
  src/wmagentattack/io_hmm_world_model.py \
  src/wmagentattack/event_world_model.py \
  src/wmagentattack/multiseed_labels.py \
  scripts/64_build_agentdojo_v2_final.py \
  scripts/110_train_io_hmm.py \
  scripts/111_train_event_world_model.py \
  scripts/112_summarize_factorized_smoke.py \
  > "$ARCHIVE/code.sha256"

if [[ ! -s "$DATA/audit.json" ]]; then
  "$PY" scripts/64_build_agentdojo_v2_final.py \
    --split-dir "$SOURCE" \
    --out-dir "$DATA" \
    --dataset-version v2.2-factorized-joint \
    --expected-attack-seed 7 \
    --expected-attack-seed 17 \
    --expected-attack-seed 29 \
    --expected-attack-seed 43 \
    --expected-attack-seed 61 \
    --expected-attack-groups 400 \
    > "$ARCHIVE/label_upgrade_stdout.json"
fi

for STATES in 4 6 8; do
  "$PY" scripts/110_train_io_hmm.py \
    --train "$DATA/train_trajectories.jsonl" \
    --validation "$DATA/val_trajectories.jsonl" \
    --test "$DATA/test_trajectories.jsonl" \
    --output-dir "$ARCHIVE/io_hmm/state$STATES" \
    --states "$STATES" \
    --iterations 50 \
    --restarts 3 \
    --seed 7 \
    --context-mode domain_attack \
    > "$ARCHIVE/io_hmm/state${STATES}_stdout.json"
done

for SEED in 7 17 29; do
  "$PY" scripts/111_train_event_world_model.py \
    --train "$DATA/train_trajectories.jsonl" \
    --validation "$DATA/val_trajectories.jsonl" \
    --test "$DATA/test_trajectories.jsonl" \
    --output-dir "$ARCHIVE/event/seed$SEED" \
    --epochs 8 \
    --batch-size 64 \
    --learning-rate 0.0003 \
    --hidden-size 64 \
    --layers 2 \
    --heads 4 \
    --seed "$SEED" \
    --device cuda \
    > "$ARCHIVE/event/seed${SEED}_stdout.json"
done

"$PY" scripts/112_summarize_factorized_smoke.py \
  --root "$ARCHIVE" \
  --output "$ARCHIVE/summary.json" \
  > "$ARCHIVE/summary_stdout.json"

find "$ARCHIVE" -type f -not -name checksums.sha256 -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$ARCHIVE/checksums.sha256"
date --iso-8601=seconds > "$ARCHIVE/COMPLETE"

