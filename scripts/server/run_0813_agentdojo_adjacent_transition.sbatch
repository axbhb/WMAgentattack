#!/usr/bin/env bash
#SBATCH --job-name=wm-adjacent-transition
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/0813_adjacent_transition_%j.out
#SBATCH --error=logs/0813_adjacent_transition_%j.err

set -euo pipefail

REPO=/share/guozhix/WMagentattack-repair-aug12
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0813/adjacent_transition/formal_v1
DATA=/share/guozhix/wmagentattack/0813/adjacent_transition/preflight_v1/a/dataset.json
AUDIT=/share/guozhix/wmagentattack/0813/adjacent_transition/preflight_v1/a/audit.json
PROTOCOL=configs/0813_agentdojo_adjacent_transition_protocol.json

cd "$REPO"
export PYTHONPATH="$REPO/src"
mkdir -p logs
if [[ -e "$ARCHIVE/COMPLETE" ]]; then
  echo "Refusing duplicate completed experiment: $ARCHIVE" >&2
  exit 3
fi
rm -rf "$ARCHIVE.tmp"
mkdir -p "$ARCHIVE.tmp/model" "$ARCHIVE.tmp/source_snapshot"
date --iso-8601=seconds > "$ARCHIVE.tmp/started_at.txt"
{
  echo "job_id=${SLURM_JOB_ID:-interactive}"
  echo "host=$(hostname)"
  echo "git_head=$(git rev-parse HEAD)"
  "$PY" --version
  "$PY" - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"device_memory={torch.cuda.get_device_properties(0).total_memory}")
PY
} | tee "$ARCHIVE.tmp/runtime.txt"

cp "$PROTOCOL" "$ARCHIVE.tmp/protocol.frozen.json"
cp "$DATA" "$ARCHIVE.tmp/data.dataset.json"
cp "$AUDIT" "$ARCHIVE.tmp/data.audit.json"
cp src/wmagentattack/adjacent_transition.py "$ARCHIVE.tmp/source_snapshot/"
cp scripts/187_build_agentdojo_adjacent_transition.py "$ARCHIVE.tmp/source_snapshot/"
cp scripts/188_train_agentdojo_adjacent_transition.py "$ARCHIVE.tmp/source_snapshot/"
cp scripts/189_summarize_agentdojo_adjacent_transition.py "$ARCHIVE.tmp/source_snapshot/"
cp scripts/server/run_0813_agentdojo_adjacent_transition.sbatch "$ARCHIVE.tmp/source_snapshot/"

sha256sum \
  "$PROTOCOL" "$DATA" "$AUDIT" \
  src/wmagentattack/adjacent_transition.py \
  scripts/187_build_agentdojo_adjacent_transition.py \
  scripts/188_train_agentdojo_adjacent_transition.py \
  scripts/189_summarize_agentdojo_adjacent_transition.py \
  scripts/server/run_0813_agentdojo_adjacent_transition.sbatch \
  > "$ARCHIVE.tmp/prerun.sha256"

"$PY" -m pytest \
  tests/test_adjacent_transition.py \
  tests/test_three_source_unified.py \
  tests/test_hybrid_semantic_world_model.py \
  -q | tee "$ARCHIVE.tmp/preflight_tests.txt"

"$PY" scripts/188_train_agentdojo_adjacent_transition.py \
  --protocol "$PROTOCOL" \
  --dataset "$DATA" \
  --audit "$AUDIT" \
  --output-dir "$ARCHIVE.tmp/model" \
  --device cuda \
  > "$ARCHIVE.tmp/training_stdout.json"

"$PY" scripts/189_summarize_agentdojo_adjacent_transition.py \
  --protocol "$PROTOCOL" \
  --dataset "$DATA" \
  --audit "$AUDIT" \
  --predictions "$ARCHIVE.tmp/model/predictions.jsonl" \
  --run-metrics "$ARCHIVE.tmp/model/run_metrics.json" \
  --output "$ARCHIVE.tmp/summary.json" \
  --markdown "$ARCHIVE.tmp/results.md" \
  > "$ARCHIVE.tmp/summary_stdout.json"

sha256sum -c "$ARCHIVE.tmp/prerun.sha256" \
  > "$ARCHIVE.tmp/prerun.postrun.verify.txt"
date --iso-8601=seconds > "$ARCHIVE.tmp/finished_at.txt"
{
  echo "new_llm_calls=0"
  echo "new_tool_executions=0"
  echo "real_external_endpoint_calls=0"
  echo "new_attack_generation=0"
  echo "dreamer_runs=0"
  echo "runtime_failures=0"
} >> "$ARCHIVE.tmp/runtime.txt"
echo "adjacent transition complete" > "$ARCHIVE.tmp/COMPLETE"
(
  cd "$ARCHIVE.tmp"
  find . -type f ! -name checksums.sha256 -print0 \
    | sort -z | xargs -0 sha256sum > checksums.sha256
)
mv "$ARCHIVE.tmp" "$ARCHIVE"
echo "AGENTDOJO_ADJACENT_TRANSITION_DONE"
