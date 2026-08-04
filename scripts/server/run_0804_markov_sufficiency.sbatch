#!/usr/bin/env bash
#SBATCH --job-name=wma-semwm-s3
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/share/guozhix/WMagentattack-autoresearch-aug4/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-autoresearch-aug4/logs/%x-%j.err

set -euo pipefail

WT=/share/guozhix/WMagentattack-autoresearch-aug4
ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
PROTOCOL=configs/0804_markov_sufficiency_protocol.json
RAW_SOURCE=/share/guozhix/wmagentattack/0729/custom_panel_v2_architecture_ablation/preflight_v2/panel_v2_architecture_dataset.json
SEMANTIC_SOURCE=/share/guozhix/wmagentattack/0804/autoresearch_semantic_wm_v3/stage1/rebuild_a/semantic_state_v3_dataset.json
ARCHIVE=/share/guozhix/wmagentattack/0804/autoresearch_semantic_wm_v3/stage3/fixed_v1

cd "$WT"
if [[ -e "$ARCHIVE" ]]; then
  echo "Refusing to overwrite frozen Stage 3 archive: $ARCHIVE" >&2
  exit 2
fi
mkdir -p "$ARCHIVE/data" "$ARCHIVE/model" "$ARCHIVE/source_snapshot" logs
export PYTHONPATH="$WT/src:$ROOT/external/agentdojo/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

failure_record() {
  status=$?
  {
    echo "failed_at=$(date --iso-8601=seconds)"
    echo "exit_status=$status"
  } > "$ARCHIVE/FAILED"
  exit "$status"
}
trap failure_record ERR

cp "$PROTOCOL" "$ARCHIVE/protocol.preregistered.json"
cp reports/0804_autoresearch_stage3_markov_sufficiency_preregistration.md "$ARCHIVE/preregistration.md"
cp "$RAW_SOURCE" "$ARCHIVE/data/raw_panel_dataset.json"
cp "$SEMANTIC_SOURCE" "$ARCHIVE/data/semantic_state_v3_dataset.json"
cp \
  src/wmagentattack/semantic_state_v3.py \
  src/wmagentattack/hybrid_semantic_world_model.py \
  src/wmagentattack/markov_sufficiency.py \
  scripts/148_train_markov_sufficiency.py \
  scripts/149_summarize_markov_sufficiency.py \
  scripts/150_audit_markov_sufficiency_preflight.py \
  scripts/server/run_0804_markov_sufficiency.sbatch \
  "$ARCHIVE/source_snapshot/"

{
  echo "job_id=$SLURM_JOB_ID"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "git_head=$(git rev-parse HEAD)"
  "$PY" --version
  echo "victim_model_calls=0"
  echo "attacks_constructed=0"
  echo "dreamer_training=0"
} > "$ARCHIVE/runtime.txt"

sha256sum \
  "$PROTOCOL" \
  reports/0804_autoresearch_stage3_markov_sufficiency_preregistration.md \
  src/wmagentattack/semantic_state_v3.py \
  src/wmagentattack/hybrid_semantic_world_model.py \
  src/wmagentattack/markov_sufficiency.py \
  scripts/148_train_markov_sufficiency.py \
  scripts/149_summarize_markov_sufficiency.py \
  scripts/150_audit_markov_sufficiency_preflight.py \
  scripts/server/run_0804_markov_sufficiency.sbatch \
  tests/test_semantic_state_v3.py \
  tests/test_hybrid_semantic_world_model.py \
  tests/test_markov_sufficiency.py \
  > "$ARCHIVE/code.prerun.sha256"

"$PY" -m pytest -q \
  tests/test_semantic_state_v3.py \
  tests/test_hybrid_semantic_world_model.py \
  tests/test_markov_sufficiency.py \
  > "$ARCHIVE/preflight_tests.txt"

"$PY" scripts/150_audit_markov_sufficiency_preflight.py \
  --protocol "$PROTOCOL" \
  --source-dataset "$ARCHIVE/data/raw_panel_dataset.json" \
  --semantic-dataset "$ARCHIVE/data/semantic_state_v3_dataset.json" \
  --output "$ARCHIVE/preflight.json" \
  > "$ARCHIVE/preflight_stdout.json"

"$PY" scripts/148_train_markov_sufficiency.py \
  --protocol "$PROTOCOL" \
  --source-dataset "$ARCHIVE/data/raw_panel_dataset.json" \
  --semantic-dataset "$ARCHIVE/data/semantic_state_v3_dataset.json" \
  --output-dir "$ARCHIVE/model" \
  --device cpu \
  > "$ARCHIVE/training_stdout.jsonl"

"$PY" scripts/149_summarize_markov_sufficiency.py \
  --protocol "$PROTOCOL" \
  --run-metrics "$ARCHIVE/model/run_metrics.json" \
  --predictions "$ARCHIVE/model/predictions.jsonl" \
  --output "$ARCHIVE/summary.json" \
  --report "$ARCHIVE/results.md" \
  > "$ARCHIVE/summary_stdout.json"

sha256sum -c "$ARCHIVE/code.prerun.sha256" > "$ARCHIVE/code.postrun.verify.txt"
{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "runtime_failures=0"
} >> "$ARCHIVE/runtime.txt"
find "$ARCHIVE" -type f \
  -not -name checksums.sha256 \
  -not -name COMPLETE \
  -not -name FAILED \
  -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$ARCHIVE/checksums.sha256"
date --iso-8601=seconds > "$ARCHIVE/COMPLETE"
trap - ERR
echo "MARKOV_SUFFICIENCY_STAGE3_DONE $(date -Is)"
