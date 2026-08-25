#!/usr/bin/env bash
#SBATCH --job-name=wma-v34-gate
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/share/guozhix/WMagentattack-paired-attack-aug26/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-paired-attack-aug26/logs/%x-%j.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack-paired-attack-aug26
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0825/paired_attack_intervention_v34/formal_v1
PROTOCOL=$ROOT/configs/0825_paired_attack_intervention_v34_protocol.json
LEGACY_MANIFEST=/share/guozhix/wmagentattack/0714/agentdojo_v2/manifests/screen_manifest.json
LEGACY_LABELS=/share/guozhix/wmagentattack/0714/agentdojo_v2/final_dataset/label_groups.jsonl

cd "$ROOT"
mkdir -p "$ARCHIVE/model" logs
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

for seed in 503 509 521; do
  test -s "$ARCHIVE/results/seed${seed}/result.json"
  test -s "$ARCHIVE/results/seed${seed}/COMPLETE"
done

"$PY" -m pytest -q tests/test_paired_attack_intervention.py tests/test_attack_conditioned_ranker.py
"$PY" scripts/281_train_paired_attack_selector_v34.py \
  --protocol "$PROTOCOL" \
  --legacy-manifest "$LEGACY_MANIFEST" \
  --legacy-label-groups "$LEGACY_LABELS" \
  --paired-manifest "$ARCHIVE/manifest.json" \
  --seed-result "$ARCHIVE/results/seed503/result.json" \
  --seed-result "$ARCHIVE/results/seed509/result.json" \
  --seed-result "$ARCHIVE/results/seed521/result.json" \
  --output-dir "$ARCHIVE/model"
"$PY" scripts/282_gate_paired_attack_selector_v34.py \
  --protocol "$PROTOCOL" \
  --predictions "$ARCHIVE/model/predictions.jsonl" \
  --data-audit "$ARCHIVE/model/data_audit.json" \
  --run-metrics "$ARCHIVE/model/run_metrics.json" \
  --output "$ARCHIVE/gate.json"

date -Iseconds > "$ARCHIVE/COMPLETE"
echo "WMA_V34_GATE_DONE $(date -Is)"
