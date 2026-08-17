#!/usr/bin/env bash
#SBATCH --job-name=wm-structured-residual-v6
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/0817_structured_residual_v6_%j.out
#SBATCH --error=logs/0817_structured_residual_v6_%j.err
set -euo pipefail
ROOT=/share/guozhix/WMagentattack-residual-v6-aug17
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
A=/share/guozhix/wmagentattack/0817/structured_residual_v6/formal_v1
DATA=/share/guozhix/wmagentattack/0817/structured_joint_outcome_v5/preflight_v1/a/dataset.json
AUDIT=/share/guozhix/wmagentattack/0817/structured_joint_outcome_v5/preflight_v1/a/audit.json
P=$ROOT/configs/0817_structured_residual_v6_protocol.json
cd "$ROOT";export PYTHONPATH="$ROOT/src";mkdir -p logs
if [[ -e "$A/COMPLETE" ]];then exit 3;fi
rm -rf "$A.tmp";mkdir -p "$A.tmp/results" "$A.tmp/source_snapshot"
echo job_id=$SLURM_JOB_ID > "$A.tmp/runtime.txt";echo git_commit=$(git rev-parse HEAD) >> "$A.tmp/runtime.txt"
cp "$P" "$A.tmp/protocol.json";cp src/wmagentattack/structured_residual_dynamics.py scripts/203_train_structured_residual_v6.py scripts/204_summarize_structured_residual_v6.py "$A.tmp/source_snapshot/"
sha256sum "$P" "$DATA" "$AUDIT" src/wmagentattack/structured_residual_dynamics.py scripts/203_train_structured_residual_v6.py scripts/204_summarize_structured_residual_v6.py > "$A.tmp/frozen.sha256"
"$PY" -m pytest tests/test_structured_residual_dynamics.py tests/test_structured_joint_model_context.py tests/test_joint_outcome_auxiliary.py -q > "$A.tmp/tests.txt"
"$PY" scripts/203_train_structured_residual_v6.py --protocol "$P" --dataset "$DATA" --audit "$AUDIT" --output-dir "$A.tmp/results"
"$PY" scripts/204_summarize_structured_residual_v6.py --protocol "$P" --predictions "$A.tmp/results/predictions.jsonl" --run-metrics "$A.tmp/results/run_metrics.json" --output "$A.tmp/results/summary.json" --markdown "$A.tmp/results/results.md"
sha256sum -c "$A.tmp/frozen.sha256" > "$A.tmp/verify.txt";touch "$A.tmp/COMPLETE";(cd "$A.tmp";find . -type f ! -name checksums.sha256 -print0|sort -z|xargs -0 sha256sum > checksums.sha256);mv "$A.tmp" "$A";echo STRUCTURED_RESIDUAL_V6_DONE
