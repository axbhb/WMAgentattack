#!/usr/bin/env bash
#SBATCH --job-name=wm-v11-factor
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=logs/0818_factorized_transition_v11_%j.out
#SBATCH --error=logs/0818_factorized_transition_v11_%j.err

set -euo pipefail
ROOT=/share/guozhix/WMagentattack-factorized-transition-aug18
ARCHIVE=/share/guozhix/wmagentattack/0818/factorized_semantic_transition_v11/stage_f2/formal_v1
TMP=${ARCHIVE}.tmp
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
SOURCE=/share/guozhix/wmagentattack/0817/structured_joint_outcome_v5/preflight_v1/a/dataset.json
FACTORS=/share/guozhix/wmagentattack/0818/factorized_semantic_transition_v11/label_gate/formal_v1/a/dataset.json
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
mkdir -p logs
if [[ -e "$ARCHIVE" || -e "$TMP" ]]; then exit 3; fi
mkdir -p "$TMP/results" "$TMP/source_snapshot"
printf 'job_id=%s\ngit_commit=%s\n' "$SLURM_JOB_ID" "$(git rev-parse HEAD)" > "$TMP/runtime.txt"
cp configs/0818_factorized_transition_v11_protocol.json "$TMP/protocol.json"
cp src/wmagentattack/factorized_transition_dynamics.py \
  src/wmagentattack/factorized_transition_labels.py \
  scripts/221_train_factorized_transition_v11.py \
  scripts/222_summarize_factorized_transition_v11.py "$TMP/source_snapshot/"
sha256sum configs/0818_factorized_transition_v11_protocol.json "$SOURCE" "$FACTORS" \
  src/wmagentattack/factorized_transition_dynamics.py \
  scripts/221_train_factorized_transition_v11.py \
  scripts/222_summarize_factorized_transition_v11.py > "$TMP/frozen.sha256"
"$PY" -m pytest tests/test_factorized_transition_labels.py \
  tests/test_factorized_transition_dynamics.py \
  tests/test_factorized_transition_v11_summary.py -q > "$TMP/tests.txt"
"$PY" scripts/221_train_factorized_transition_v11.py \
  --protocol configs/0818_factorized_transition_v11_protocol.json \
  --dataset "$SOURCE" --factor-dataset "$FACTORS" \
  --output-dir "$TMP/results" --device cpu
"$PY" scripts/222_summarize_factorized_transition_v11.py \
  --protocol configs/0818_factorized_transition_v11_protocol.json \
  --predictions "$TMP/results/predictions.jsonl" \
  --run-metrics "$TMP/results/run_metrics.json" \
  --output "$TMP/stage_f2_gate.json" \
  --markdown "$TMP/stage_f2_results.md"
sha256sum -c "$TMP/frozen.sha256" > "$TMP/verify.txt"
touch "$TMP/COMPLETE"
(cd "$TMP"; find . -type f ! -name checksums.sha256 -print0 | sort -z | xargs -0 sha256sum > checksums.sha256)
mv "$TMP" "$ARCHIVE"
echo FACTORIZED_TRANSITION_V11_DONE
