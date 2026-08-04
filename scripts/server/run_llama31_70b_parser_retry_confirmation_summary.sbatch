#!/usr/bin/env bash
#SBATCH --job-name=wma-parser-confirm-sum
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%j.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
DEV=/share/guozhix/wmagentattack/0716/function_tag_parser_retry/function_tag_parser_retry_20260716_v1
ARCHIVE=/share/guozhix/wmagentattack/0716/function_tag_parser_retry_confirmation/function_tag_parser_retry_confirmation_20260716_v1
PROTOCOL=$ROOT/configs/0716_function_tag_parser_retry_confirmation.json

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/external/agentdojo/src:${PYTHONPATH:-}"
"$PY" scripts/102_summarize_fresh_clean_multiseed.py --manifest "$ARCHIVE/travel_manifest.json" --archive-root "$ARCHIVE" --seeds 109 113 127 --chunks 2 --output "$ARCHIVE/confirmation_multiseed_summary.json" > "$ARCHIVE/confirmation_multiseed_summary_stdout.json"
"$PY" scripts/103_audit_fresh_clean_failures.py --archive-root "$ARCHIVE" --seeds 109 113 127 --chunks 2 --output "$ARCHIVE/confirmation_failure_audit.json" > "$ARCHIVE/confirmation_failure_audit_stdout.json"
"$PY" scripts/108_evaluate_parser_retry_confirmation.py --development-summary "$DEV/parser_retry_travel_multiseed_summary.json" --confirmation-summary "$ARCHIVE/confirmation_multiseed_summary.json" --failure-audit "$ARCHIVE/confirmation_failure_audit.json" --protocol "$PROTOCOL" --output "$ARCHIVE/confirmation_gate.json" > "$ARCHIVE/confirmation_gate_stdout.json"
cp -p "$PROTOCOL" "$ARCHIVE/frozen_confirmation_protocol.json"

echo "AGENTDOJO_PARSER_CONFIRM_SUMMARY_DONE $(date -Is)"
