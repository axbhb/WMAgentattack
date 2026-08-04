#!/usr/bin/env bash
#SBATCH --job-name=wma-panel-v2-sum
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%j.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0728/custom_clean_panel_v2/fixed_v1
FROZEN=$ARCHIVE/frozen_inputs

cd "$ROOT"
mkdir -p "$ARCHIVE/ledger_gold" logs
(
  cd "$FROZEN"
  sha256sum --quiet -c SHA256SUMS
)
sha256sum --quiet -c "$FROZEN/frozen_repo_SHA256SUMS"
export PYTHONPATH="$ROOT/src:$ROOT/external/agentdojo/src:${PYTHONPATH:-}"

"$PY" -m pytest \
  tests/test_custom_agentdojo_panel_v2.py \
  tests/test_custom_clean_panel_v2_manifest.py \
  tests/test_custom_clean_panel_v2_summary.py \
  tests/test_factorized_evaluator_v2.py \
  tests/test_trace_execution_pairing.py \
  tests/test_structured_ledger_v2.py \
  tests/test_ledger_v2_gold_fixtures.py \
  tests/test_label_blind_ledger_v2_fixtures.py \
  -q --junitxml="$ARCHIVE/clean_ledger_tests.xml"

"$PY" scripts/128_audit_ledger_v2_gold_fixtures.py \
  --fixtures configs/0726_travel_ledger_v2_gold_fixtures.json \
  --registry configs/0726_travel_tool_output_adapter_registry_v2.json \
  --test-junit "$ARCHIVE/clean_ledger_tests.xml" \
  --output-dir "$ARCHIVE/ledger_gold" \
  > "$ARCHIVE/ledger_gold/stdout.json"

"$PY" scripts/141_summarize_custom_clean_panel_v2.py \
  --protocol "$FROZEN/protocol.json" \
  --greedy-manifest "$FROZEN/greedy_manifest.json" \
  --stochastic-manifest "$FROZEN/stochastic_manifest.json" \
  --contracts "$FROZEN/contracts.json" \
  --run-plan "$FROZEN/run_plan.json" \
  --aliases "$FROZEN/aliases.json" \
  --preflight "$FROZEN/preflight.json" \
  --ledger-audit "$ARCHIVE/ledger_gold/audit.json" \
  --archive-root "$ARCHIVE" \
  --output "$ARCHIVE/custom_clean_panel_v2_summary.json" \
  > "$ARCHIVE/custom_clean_panel_v2_summary_stdout.json"

sha256sum \
  "$ARCHIVE/custom_clean_panel_v2_summary.json" \
  "$ARCHIVE/custom_clean_panel_v2_summary_stdout.json" \
  "$ARCHIVE/ledger_gold/audit.json" \
  "$ARCHIVE/clean_ledger_tests.xml" \
  > "$ARCHIVE/result_SHA256SUMS"

echo "CUSTOM_CLEAN_PANEL_V2_SUMMARY_DONE $(date -Is)"
