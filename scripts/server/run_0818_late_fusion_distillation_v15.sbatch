#!/usr/bin/env bash
#SBATCH --job-name=wm-v15-distill
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=logs/0818_late_fusion_distillation_v15_%j.out
#SBATCH --error=logs/0818_late_fusion_distillation_v15_%j.err
set -euo pipefail
ROOT=/share/guozhix/WMagentattack-late-fusion-aug18;A=/share/guozhix/wmagentattack/0818/late_fusion_distillation_v15/formal_v1;TMP=${A}.tmp;PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
EVENTS=/share/guozhix/wmagentattack/0817/structured_joint_outcome_v5/preflight_v1/a/dataset.json;GRAPH=/share/guozhix/wmagentattack/0818/action_event_graph_v12/data_gate/formal_v1/a/dataset.json;PARTITION=/share/guozhix/wmagentattack/0818/hybrid_evidence_residual_v14/data_gate/formal_v1/a/dataset.json
cd "$ROOT";export PYTHONPATH="$ROOT/src";mkdir -p logs;if [[ -e "$A" || -e "$TMP" ]];then exit 3;fi;mkdir -p "$TMP/results" "$TMP/source_snapshot";printf 'job_id=%s\ngit_commit=%s\n' "$SLURM_JOB_ID" "$(git rev-parse HEAD)" > "$TMP/runtime.txt";cp configs/0818_late_fusion_distillation_v15_protocol.json "$TMP/protocol.json";cp src/wmagentattack/late_fusion_distilled_dynamics.py scripts/232_train_late_fusion_distillation_v15.py scripts/233_summarize_late_fusion_distillation_v15.py "$TMP/source_snapshot/"
sha256sum configs/0818_late_fusion_distillation_v15_protocol.json "$EVENTS" "$GRAPH" "$PARTITION" src/wmagentattack/late_fusion_distilled_dynamics.py scripts/232_train_late_fusion_distillation_v15.py scripts/233_summarize_late_fusion_distillation_v15.py > "$TMP/frozen.sha256"
"$PY" -m pytest tests/test_late_fusion_distilled_dynamics.py tests/test_late_fusion_distillation_v15_summary.py -q > "$TMP/tests.txt"
"$PY" scripts/232_train_late_fusion_distillation_v15.py --protocol configs/0818_late_fusion_distillation_v15_protocol.json --events "$EVENTS" --graph-dataset "$GRAPH" --partition "$PARTITION" --output-dir "$TMP/results" --device cpu
"$PY" scripts/233_summarize_late_fusion_distillation_v15.py --protocol configs/0818_late_fusion_distillation_v15_protocol.json --predictions "$TMP/results/predictions.jsonl" --run-metrics "$TMP/results/run_metrics.json" --composite "$TMP/results/composite_predictions.jsonl" --output "$TMP/distillation_gate.json" --markdown "$TMP/distillation_results.md"
sha256sum -c "$TMP/frozen.sha256" > "$TMP/verify.txt";touch "$TMP/COMPLETE";(cd "$TMP";find . -type f ! -name checksums.sha256 -print0|sort -z|xargs -0 sha256sum > checksums.sha256);mv "$TMP" "$A";echo LATE_FUSION_DISTILLATION_V15_DONE
