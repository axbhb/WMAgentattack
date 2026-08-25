#!/usr/bin/env bash
#SBATCH --job-name=wma-v33-attack-rank
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/share/guozhix/WMagentattack-attack-ranking-aug25/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-attack-ranking-aug25/logs/%x-%j.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack-attack-ranking-aug25
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
PROTOCOL=$ROOT/configs/0825_attack_conditioned_ranking_v33_protocol.json
ARCHIVE=/share/guozhix/wmagentattack/0825/attack_conditioned_ranking_v33/formal_v1
MANIFEST=/share/guozhix/wmagentattack/0714/agentdojo_v2/manifests/screen_manifest.json
LABELS=/share/guozhix/wmagentattack/0714/agentdojo_v2/final_dataset/label_groups.jsonl
V5=/share/guozhix/wmagentattack/0817/structured_joint_outcome_v5/formal_v1/results/predictions.jsonl

cd "$ROOT"
mkdir -p "$ARCHIVE/results" logs
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

"$PY" -m pytest -q tests/test_attack_conditioned_ranker.py
"$PY" scripts/277_train_attack_conditioned_ranking_v33.py \
  --protocol "$PROTOCOL" \
  --manifest "$MANIFEST" \
  --label-groups "$LABELS" \
  --v5-predictions "$V5" \
  --output-dir "$ARCHIVE/results"
"$PY" scripts/278_gate_attack_conditioned_ranking_v33.py \
  --protocol "$PROTOCOL" \
  --predictions "$ARCHIVE/results/predictions.jsonl" \
  --run-metrics "$ARCHIVE/results/run_metrics.json" \
  --output "$ARCHIVE/gate.json"

echo "WMA_V33_ATTACK_RANK_DONE $(date -Is)"
