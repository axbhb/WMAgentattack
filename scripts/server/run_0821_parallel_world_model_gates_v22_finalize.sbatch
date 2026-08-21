#!/usr/bin/env bash
#SBATCH --job-name=wma-v22-finalize
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/share/guozhix/WMagentattack-parallel-gates-aug21/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-parallel-gates-aug21/logs/%x-%j.err

set -euo pipefail
WT=/share/guozhix/WMagentattack-parallel-gates-aug21
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
PROTOCOL=configs/0821_parallel_world_model_gates_v22_protocol.json
ROOT=/share/guozhix/wmagentattack/0821/parallel_world_model_gates_v22/formal_v1
OPEN=$ROOT/open_vocabulary/results
LONG=$ROOT/long_horizon/results
V4=/share/guozhix/wmagentattack/0817/factorized_belief_wm_v4/pilot_v1/results/predictions.jsonl
cd "$WT"
if [[ -e "$ROOT/COMPLETE" ]]; then echo duplicate-finalize-v22 >&2; exit 3; fi
test -e "$ROOT/open_vocabulary/COMPLETE"
test -e "$ROOT/long_horizon/COMPLETE"
mkdir -p "$ROOT/final"
"$PY" scripts/253_gate_data_generation_design_v22.py \
  --protocol "$PROTOCOL" --output "$ROOT/final/data_generation_design_gate.json" \
  > "$ROOT/final/data_generation_design_stdout.json"
"$PY" scripts/252_gate_long_horizon_controls_v22.py \
  --protocol "$PROTOCOL" --data-gate "$LONG/data_gate.json" \
  --metrics "$LONG/run_metrics.json" --predictions "$LONG/predictions.jsonl" \
  --v4-predictions "$V4" --open-gate "$OPEN/gate.json" \
  --output "$ROOT/final/long_horizon_gate.json" > "$ROOT/final/long_horizon_gate_stdout.json"
"$PY" scripts/254_finalize_parallel_world_model_gates_v22.py \
  --open-gate "$OPEN/gate.json" --long-gate "$ROOT/final/long_horizon_gate.json" \
  --data-design-gate "$ROOT/final/data_generation_design_gate.json" \
  --output "$ROOT/final/final_gate.json" \
  --markdown "$ROOT/final/results.md" > "$ROOT/final/finalize_stdout.json"
find "$ROOT" -type f -not -name checksums.sha256 -not -name COMPLETE -print0 \
  | sort -z | xargs -0 sha256sum > "$ROOT/checksums.sha256"
date -Is > "$ROOT/COMPLETE"

