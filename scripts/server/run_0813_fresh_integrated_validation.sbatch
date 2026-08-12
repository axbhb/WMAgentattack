#!/usr/bin/env bash
#SBATCH --job-name=wma-integrated-v1
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/0813_integrated_v1_%j.out
#SBATCH --error=logs/0813_integrated_v1_%j.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack-fresh-validation-aug13
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0813/fresh_integrated_validation/formal_v1
DATA=$ARCHIVE/integrated_dataset/a/dataset.json
AUDIT=$ARCHIVE/integrated_dataset/a/audit.json
PROTOCOL=configs/0813_fresh_integrated_validation_protocol.json

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/external/agentdojo/src:${PYTHONPATH:-}"
mkdir -p logs
test -e "$ARCHIVE/fresh_dataset/clean_gate.json"
test ! -e "$ARCHIVE/model/COMPLETE"
rm -rf "$ARCHIVE/model.tmp"
mkdir -p "$ARCHIVE/model.tmp/source_snapshot"
cp "$PROTOCOL" "$ARCHIVE/model.tmp/protocol.frozen.json"
cp src/wmagentattack/fresh_integrated_validation.py scripts/194_build_fresh_integrated_validation_dataset.py scripts/195_train_fresh_integrated_validation.py scripts/196_summarize_fresh_integrated_validation.py scripts/server/run_0813_fresh_integrated_validation.sbatch "$ARCHIVE/model.tmp/source_snapshot/"
sha256sum "$PROTOCOL" "$DATA" "$AUDIT" src/wmagentattack/fresh_integrated_validation.py scripts/195_train_fresh_integrated_validation.py scripts/196_summarize_fresh_integrated_validation.py > "$ARCHIVE/model.tmp/prerun.sha256"
"$PY" -m pytest tests/test_fresh_integrated_validation.py tests/test_source_residual_adapter.py tests/test_adjacent_transition.py -q | tee "$ARCHIVE/model.tmp/preflight_tests.txt"
"$PY" -c 'import torch; print(torch.cuda.get_device_name(0))' | tee "$ARCHIVE/model.tmp/device.txt"
"$PY" scripts/195_train_fresh_integrated_validation.py \
  --protocol "$PROTOCOL" \
  --dataset "$DATA" \
  --audit "$AUDIT" \
  --output-dir "$ARCHIVE/model.tmp/run" \
  --device cuda > "$ARCHIVE/model.tmp/training_stdout.json"
"$PY" scripts/196_summarize_fresh_integrated_validation.py \
  --protocol "$PROTOCOL" \
  --predictions "$ARCHIVE/model.tmp/run/predictions.jsonl" \
  --run-metrics "$ARCHIVE/model.tmp/run/run_metrics.json" \
  --output "$ARCHIVE/model.tmp/summary.json" \
  --markdown "$ARCHIVE/model.tmp/results.md" > "$ARCHIVE/model.tmp/summary_stdout.json"
sha256sum -c "$ARCHIVE/model.tmp/prerun.sha256" > "$ARCHIVE/model.tmp/prerun.postrun.verify.txt"
printf 'new_llm_calls=0\nnew_tool_executions=0\nreal_external_endpoint_calls=0\nattack_episodes=0\ndreamer_runs=0\nruntime_failures=0\n' > "$ARCHIVE/model.tmp/runtime_contract.txt"
date --iso-8601=seconds > "$ARCHIVE/model.tmp/finished_at.txt"
echo complete > "$ARCHIVE/model.tmp/COMPLETE"
(cd "$ARCHIVE/model.tmp" && find . -type f ! -name checksums.sha256 -print0 | sort -z | xargs -0 sha256sum > checksums.sha256)
mv "$ARCHIVE/model.tmp" "$ARCHIVE/model"
echo "FRESH_INTEGRATED_VALIDATION_DONE $(date -Is)"
