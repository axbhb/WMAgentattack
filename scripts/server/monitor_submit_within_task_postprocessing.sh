#!/usr/bin/env bash

set -u

ROOT=/share/guozhix/WMagentattack
ARCHIVE=/share/guozhix/wmagentattack/0712/within_task_contrast
LOCK="$ARCHIVE/postprocessing_submit.lock"
MANIFEST="$ARCHIVE/postprocessing_job_manifest.tsv"
REPLAY_DEPENDENCY=4268:4272:4273:4274:4275
SCORE_SCRIPT=scripts/server/run_score_injection_conditioned_contrast.sbatch

mkdir -p "$ARCHIVE"
cd "$ROOT" || exit 2
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "POSTPROCESSING_MONITOR_ALREADY_RUNNING $(date -Is)"
  exit 0
fi
touch "$MANIFEST"

manifest_id() {
  awk -F '\t' -v kind="$1" '$1 == kind {value=$2} END {print value}' "$MANIFEST"
}

queued_id() {
  squeue -h -u "$USER" -n "$1" -o '%A' 2>/dev/null | awk 'NF {print $1; exit}'
}

record_once() {
  local kind="$1"
  local job_id="$2"
  if [[ -z "$(manifest_id "$kind")" ]]; then
    printf '%s\t%s\t%s\n' "$kind" "$job_id" "$(date -Is)" >>"$MANIFEST"
  fi
}

submit_with_timeout() {
  local output
  output=$(timeout 15s sbatch --parsable "${@:2}" "$1" 2>&1)
  local status=$?
  if (( status == 0 )); then
    printf '%s\n' "$output" | tail -n 1
    return 0
  fi
  echo "SBATCH_RETRY status=$status message=$output $(date -Is)" >&2
  return "$status"
}

deadline=$((SECONDS + 21600))
while (( SECONDS < deadline )); do
  score_id=$(manifest_id injection_score)
  if [[ -z "$score_id" ]]; then
    score_id=$(queued_id wma-score-ic)
    if [[ -n "$score_id" ]]; then
      record_once injection_score "$score_id"
    else
      score_id=$(submit_with_timeout "$SCORE_SCRIPT" \
        --dependency="afterok:$REPLAY_DEPENDENCY") || score_id=""
      if [[ -n "$score_id" ]]; then
        record_once injection_score "$score_id"
        echo "INJECTION_SCORE_SUBMITTED job_id=$score_id $(date -Is)"
      fi
    fi
  fi

  if [[ -n "$score_id" ]]; then
    evaluation_id=$(manifest_id main_evaluation)
    if [[ -z "$evaluation_id" ]]; then
      evaluation_id=$(queued_id wma-contrast-eval)
      if [[ -z "$evaluation_id" ]]; then
        evaluation_id=$(submit_with_timeout \
          scripts/server/run_merge_and_evaluate_within_task_contrast.sh \
          --dependency="afterok:$REPLAY_DEPENDENCY:$score_id" \
          --partition=6000ada \
          --cpus-per-task=8 \
          --mem=32G \
          --time=02:00:00 \
          --job-name=wma-contrast-eval \
          --output="$ROOT/logs/wma-contrast-eval-%j.out" \
          --error="$ROOT/logs/wma-contrast-eval-%j.err") || evaluation_id=""
      fi
      if [[ -n "$evaluation_id" ]]; then
        record_once main_evaluation "$evaluation_id"
        echo "MAIN_EVALUATION_SUBMITTED job_id=$evaluation_id $(date -Is)"
      fi
    fi
  fi

  if [[ -n "$(manifest_id injection_score)" && -n "$(manifest_id main_evaluation)" ]]; then
    echo "POSTPROCESSING_MONITOR_DONE $(date -Is)"
    exit 0
  fi
  sleep 60
done

echo "POSTPROCESSING_MONITOR_TIMEOUT $(date -Is)" >&2
exit 3
