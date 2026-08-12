#!/usr/bin/env bash
#SBATCH --job-name=wm-source-head
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/0814_source_head_%j.out
#SBATCH --error=logs/0814_source_head_%j.err

set -euo pipefail
REPO=/share/guozhix/WMagentattack-repair-aug12
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0814/source_specific_head/formal_v1
DATA=/share/guozhix/wmagentattack/0812/three_source_unified_action/preflight_v1/a/dataset.json
AUDIT=/share/guozhix/wmagentattack/0812/three_source_unified_action/preflight_v1/a/audit.json
PARENT=/share/guozhix/wmagentattack/0812/three_source_unified_action/formal_v1/model/predictions.jsonl
PROTOCOL=configs/0814_source_specific_action_head_protocol.json
cd "$REPO"; export PYTHONPATH="$REPO/src"; mkdir -p logs
if [[ -e "$ARCHIVE/COMPLETE" ]]; then echo duplicate >&2; exit 3; fi
rm -rf "$ARCHIVE.tmp"; mkdir -p "$ARCHIVE.tmp/model" "$ARCHIVE.tmp/source_snapshot"
date --iso-8601=seconds > "$ARCHIVE.tmp/started_at.txt"
{ echo "job_id=${SLURM_JOB_ID:-interactive}"; echo "host=$(hostname)"; echo "git_head=$(git rev-parse HEAD)"; "$PY" --version; "$PY" -c 'import torch; print(f"torch={torch.__version__}\ndevice={torch.cuda.get_device_name(0)}")'; } | tee "$ARCHIVE.tmp/runtime.txt"
cp "$PROTOCOL" "$ARCHIVE.tmp/protocol.frozen.json"
cp src/wmagentattack/source_residual_adapter.py scripts/190_train_source_residual_adapter.py scripts/191_summarize_source_residual_adapter.py scripts/server/run_0814_source_specific_action_head.sbatch "$ARCHIVE.tmp/source_snapshot/"
sha256sum "$PROTOCOL" "$DATA" "$AUDIT" "$PARENT" src/wmagentattack/source_residual_adapter.py scripts/190_train_source_residual_adapter.py scripts/191_summarize_source_residual_adapter.py scripts/server/run_0814_source_specific_action_head.sbatch > "$ARCHIVE.tmp/prerun.sha256"
"$PY" -m pytest tests/test_source_residual_adapter.py tests/test_three_source_unified.py tests/test_hybrid_semantic_world_model.py -q | tee "$ARCHIVE.tmp/preflight_tests.txt"
"$PY" scripts/190_train_source_residual_adapter.py --protocol "$PROTOCOL" --dataset "$DATA" --audit "$AUDIT" --output-dir "$ARCHIVE.tmp/model" --device cuda > "$ARCHIVE.tmp/training_stdout.json"
"$PY" scripts/191_summarize_source_residual_adapter.py --protocol "$PROTOCOL" --predictions "$ARCHIVE.tmp/model/predictions.jsonl" --run-metrics "$ARCHIVE.tmp/model/run_metrics.json" --parent-predictions "$PARENT" --output "$ARCHIVE.tmp/summary.json" --markdown "$ARCHIVE.tmp/results.md" > "$ARCHIVE.tmp/summary_stdout.json"
sha256sum -c "$ARCHIVE.tmp/prerun.sha256" > "$ARCHIVE.tmp/prerun.postrun.verify.txt"
date --iso-8601=seconds > "$ARCHIVE.tmp/finished_at.txt"; printf '%s\n' new_llm_calls=0 new_tool_executions=0 real_external_endpoint_calls=0 new_attack_generation=0 dreamer_runs=0 runtime_failures=0 >> "$ARCHIVE.tmp/runtime.txt"
echo complete > "$ARCHIVE.tmp/COMPLETE"; (cd "$ARCHIVE.tmp"; find . -type f ! -name checksums.sha256 -print0 | sort -z | xargs -0 sha256sum > checksums.sha256); mv "$ARCHIVE.tmp" "$ARCHIVE"; echo SOURCE_SPECIFIC_HEAD_DONE
