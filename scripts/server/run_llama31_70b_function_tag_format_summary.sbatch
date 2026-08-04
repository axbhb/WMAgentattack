#!/usr/bin/env bash
#SBATCH --job-name=wma-format-travel-sum
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%j.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
BASE=/share/guozhix/wmagentattack/0716/fresh_task_screen/fresh_task_screen_20260716_v1
ARCHIVE=/share/guozhix/wmagentattack/0716/function_tag_format/function_tag_format_20260716_v1

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/external/agentdojo/src:${PYTHONPATH:-}"
"$PY" scripts/102_summarize_fresh_clean_multiseed.py \
  --manifest "$ARCHIVE/travel_manifest.json" \
  --archive-root "$ARCHIVE" \
  --seeds 101 103 107 \
  --chunks 2 \
  --output "$ARCHIVE/format_travel_multiseed_summary.json" \
  > "$ARCHIVE/format_travel_multiseed_summary_stdout.json"

"$PY" scripts/103_audit_fresh_clean_failures.py \
  --archive-root "$ARCHIVE" \
  --seeds 101 103 107 \
  --chunks 2 \
  --output "$ARCHIVE/format_travel_failure_audit.json" \
  > "$ARCHIVE/format_travel_failure_audit_stdout.json"

"$PY" scripts/105_compare_clean_tool_protocols.py \
  --baseline-archive "$BASE" \
  --candidate-archive "$ARCHIVE" \
  --baseline-chunks 4 \
  --candidate-chunks 2 \
  --seeds 101 103 107 \
  --output "$ARCHIVE/format_vs_base_function_tags_comparison.json" \
  > "$ARCHIVE/format_vs_base_function_tags_comparison_stdout.json"

echo "AGENTDOJO_FORMAT_TRAVEL_SUMMARY_DONE $(date -Is)"
