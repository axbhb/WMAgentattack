#!/usr/bin/env bash
#SBATCH --job-name=wm-v16-retrieve
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=logs/0818_retrieval_support_gate_v16_%j.out
#SBATCH --error=logs/0818_retrieval_support_gate_v16_%j.err
set -euo pipefail
ROOT=/share/guozhix/WMagentattack-retrieval-aug18;A=/share/guozhix/wmagentattack/0818/retrieval_world_model_v16/support_gate/formal_v1;TMP=${A}.tmp;PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python;EVENTS=/share/guozhix/wmagentattack/0817/structured_joint_outcome_v5/preflight_v1/a/dataset.json;GRAPH=/share/guozhix/wmagentattack/0818/action_event_graph_v12/data_gate/formal_v1/a/dataset.json
cd "$ROOT";export PYTHONPATH="$ROOT/src";mkdir -p logs;if [[ -e "$A" || -e "$TMP" ]];then exit 3;fi;mkdir -p "$TMP/results" "$TMP/source_snapshot";printf 'job_id=%s\ngit_commit=%s\n' "$SLURM_JOB_ID" "$(git rev-parse HEAD)" > "$TMP/runtime.txt";cp configs/0818_retrieval_world_model_v16_protocol.json "$TMP/protocol.json";cp scripts/234_run_retrieval_support_gate_v16.py scripts/235_summarize_retrieval_support_gate_v16.py "$TMP/source_snapshot/";sha256sum configs/0818_retrieval_world_model_v16_protocol.json "$EVENTS" "$GRAPH" scripts/234_run_retrieval_support_gate_v16.py scripts/235_summarize_retrieval_support_gate_v16.py > "$TMP/frozen.sha256"
"$PY" -m pytest tests/test_retrieval_world_model_v16.py tests/test_retrieval_world_model_v16_summary.py -q > "$TMP/tests.txt";"$PY" scripts/234_run_retrieval_support_gate_v16.py --protocol configs/0818_retrieval_world_model_v16_protocol.json --events "$EVENTS" --graph-dataset "$GRAPH" --output-dir "$TMP/results";"$PY" scripts/235_summarize_retrieval_support_gate_v16.py --protocol configs/0818_retrieval_world_model_v16_protocol.json --predictions "$TMP/results/predictions.jsonl" --run-metrics "$TMP/results/run_metrics.json" --output "$TMP/retrieval_gate.json" --markdown "$TMP/retrieval_results.md";sha256sum -c "$TMP/frozen.sha256" > "$TMP/verify.txt";touch "$TMP/COMPLETE";(cd "$TMP";find . -type f ! -name checksums.sha256 -print0|sort -z|xargs -0 sha256sum > checksums.sha256);mv "$TMP" "$A";echo RETRIEVAL_SUPPORT_GATE_V16_DONE
