#!/usr/bin/env bash
#SBATCH --job-name=wm-structured-joint-v5
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/0817_structured_joint_v5_%j.out
#SBATCH --error=logs/0817_structured_joint_v5_%j.err
set -euo pipefail
ROOT=/share/guozhix/WMagentattack-joint-v5-aug17
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0817/structured_joint_outcome_v5/formal_v1
DATA=/share/guozhix/wmagentattack/0817/structured_joint_outcome_v5/preflight_v1/a/dataset.json
AUDIT=/share/guozhix/wmagentattack/0817/structured_joint_outcome_v5/preflight_v1/a/audit.json
PROTOCOL=$ROOT/configs/0817_structured_joint_outcome_v5_protocol.json
cd "$ROOT"; export PYTHONPATH="$ROOT/src"; mkdir -p logs
if [[ -e "$ARCHIVE/COMPLETE" ]]; then exit 3; fi
rm -rf "$ARCHIVE.tmp"; mkdir -p "$ARCHIVE.tmp/results" "$ARCHIVE.tmp/source_snapshot"
echo "job_id=$SLURM_JOB_ID" > "$ARCHIVE.tmp/runtime.txt"; echo "git_commit=$(git rev-parse HEAD)" >> "$ARCHIVE.tmp/runtime.txt"; echo runtime_device=cpu >> "$ARCHIVE.tmp/runtime.txt"
cp "$PROTOCOL" "$ARCHIVE.tmp/protocol.frozen.json"
cp src/wmagentattack/joint_outcome_auxiliary.py scripts/201_train_structured_joint_outcome_v5.py scripts/202_summarize_structured_joint_outcome_v5.py "$ARCHIVE.tmp/source_snapshot/"
sha256sum "$PROTOCOL" "$DATA" "$AUDIT" src/wmagentattack/joint_outcome_auxiliary.py scripts/201_train_structured_joint_outcome_v5.py scripts/202_summarize_structured_joint_outcome_v5.py > "$ARCHIVE.tmp/frozen_inputs.sha256"
"$PY" -m pytest tests/test_joint_outcome_auxiliary.py tests/test_adjacent_transition.py -q > "$ARCHIVE.tmp/tests.txt"
"$PY" scripts/201_train_structured_joint_outcome_v5.py --protocol "$PROTOCOL" --dataset "$DATA" --audit "$AUDIT" --output-dir "$ARCHIVE.tmp/results" --device cpu
"$PY" scripts/202_summarize_structured_joint_outcome_v5.py --protocol "$PROTOCOL" --dataset "$DATA" --audit "$AUDIT" --predictions "$ARCHIVE.tmp/results/predictions.jsonl" --run-metrics "$ARCHIVE.tmp/results/run_metrics.json" --output "$ARCHIVE.tmp/results/summary.json" --markdown "$ARCHIVE.tmp/results/results.md"
sha256sum -c "$ARCHIVE.tmp/frozen_inputs.sha256" > "$ARCHIVE.tmp/postrun_verify.txt"
touch "$ARCHIVE.tmp/COMPLETE"; (cd "$ARCHIVE.tmp"; find . -type f ! -name checksums.sha256 -print0 | sort -z | xargs -0 sha256sum > checksums.sha256); mv "$ARCHIVE.tmp" "$ARCHIVE"
echo STRUCTURED_JOINT_V5_DONE
