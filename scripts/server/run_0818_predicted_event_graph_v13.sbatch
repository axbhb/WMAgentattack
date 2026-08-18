#!/usr/bin/env bash
#SBATCH --job-name=wm-v13-pgraph
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=logs/0818_predicted_event_graph_v13_%j.out
#SBATCH --error=logs/0818_predicted_event_graph_v13_%j.err

set -euo pipefail
ROOT=/share/guozhix/WMagentattack-modular-event-wm-aug18
A=/share/guozhix/wmagentattack/0818/modular_event_world_model_v13/stage_m2/formal_v1
TMP=${A}.tmp
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
EVENTS=/share/guozhix/wmagentattack/0817/structured_joint_outcome_v5/preflight_v1/a/dataset.json
GRAPH=/share/guozhix/wmagentattack/0818/action_event_graph_v12/data_gate/formal_v1/a/dataset.json
cd "$ROOT"; export PYTHONPATH="$ROOT/src"; mkdir -p logs
if [[ -e "$A" || -e "$TMP" ]]; then exit 3; fi
mkdir -p "$TMP/results" "$TMP/source_snapshot"
printf 'job_id=%s\ngit_commit=%s\n' "$SLURM_JOB_ID" "$(git rev-parse HEAD)" > "$TMP/runtime.txt"
cp configs/0818_modular_event_world_model_v13_protocol.json "$TMP/protocol.json"
cp src/wmagentattack/predicted_event_graph_dynamics.py \
  scripts/227_train_predicted_event_graph_v13.py scripts/228_summarize_predicted_event_graph_v13.py \
  "$TMP/source_snapshot/"
sha256sum configs/0818_modular_event_world_model_v13_protocol.json "$EVENTS" "$GRAPH" \
  src/wmagentattack/predicted_event_graph_dynamics.py \
  scripts/227_train_predicted_event_graph_v13.py scripts/228_summarize_predicted_event_graph_v13.py > "$TMP/frozen.sha256"
"$PY" -m pytest tests/test_predicted_event_graph_dynamics.py tests/test_predicted_event_graph_v13_summary.py -q > "$TMP/tests.txt"
"$PY" scripts/227_train_predicted_event_graph_v13.py \
  --protocol configs/0818_modular_event_world_model_v13_protocol.json \
  --events "$EVENTS" --graph-dataset "$GRAPH" --output-dir "$TMP/results" --device cpu
"$PY" scripts/228_summarize_predicted_event_graph_v13.py \
  --protocol configs/0818_modular_event_world_model_v13_protocol.json \
  --predictions "$TMP/results/predictions.jsonl" --run-metrics "$TMP/results/run_metrics.json" \
  --composite-predictions "$TMP/results/composite_predictions.jsonl" \
  --output "$TMP/predicted_graph_gate.json" --markdown "$TMP/predicted_graph_results.md"
sha256sum -c "$TMP/frozen.sha256" > "$TMP/verify.txt"
touch "$TMP/COMPLETE"
(cd "$TMP"; find . -type f ! -name checksums.sha256 -print0 | sort -z | xargs -0 sha256sum > checksums.sha256)
mv "$TMP" "$A"
echo PREDICTED_EVENT_GRAPH_V13_DONE
