#!/usr/bin/env bash
#SBATCH --job-name=wm-rel-slot-a
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/0818_relational_slot_stage_a_%j.out
#SBATCH --error=logs/0818_relational_slot_stage_a_%j.err
set -euo pipefail
ROOT=/share/guozhix/WMagentattack-relational-jepa-aug18
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
ARCH=/share/guozhix/wmagentattack/0818/relational_slot_jepa_v7/stage_a/formal_v1
DATA=/share/guozhix/wmagentattack/0817/structured_joint_outcome_v5/preflight_v1/a/dataset.json
AUDIT=/share/guozhix/wmagentattack/0817/structured_joint_outcome_v5/preflight_v1/a/audit.json
PROTOCOL=$ROOT/configs/0818_relational_slot_autoresearch_protocol.json
cd "$ROOT";export PYTHONPATH="$ROOT/src";mkdir -p logs
if [[ -e "$ARCH/COMPLETE" ]];then exit 3;fi
rm -rf "$ARCH.tmp";mkdir -p "$ARCH.tmp/results" "$ARCH.tmp/source_snapshot"
printf 'job_id=%s\ngit_commit=%s\n' "$SLURM_JOB_ID" "$(git rev-parse HEAD)" > "$ARCH.tmp/runtime.txt"
cp "$PROTOCOL" "$ARCH.tmp/protocol.json"
cp src/wmagentattack/relational_slot_latent.py scripts/205_train_relational_slot_stage_a.py scripts/206_summarize_relational_slot_stage_a.py "$ARCH.tmp/source_snapshot/"
sha256sum "$PROTOCOL" "$DATA" "$AUDIT" src/wmagentattack/relational_slot_latent.py scripts/205_train_relational_slot_stage_a.py scripts/206_summarize_relational_slot_stage_a.py > "$ARCH.tmp/frozen.sha256"
"$PY" -m pytest tests/test_relational_slot_latent.py tests/test_structured_residual_dynamics.py tests/test_joint_outcome_auxiliary.py -q > "$ARCH.tmp/tests.txt"
"$PY" scripts/205_train_relational_slot_stage_a.py --protocol "$PROTOCOL" --dataset "$DATA" --audit "$AUDIT" --output-dir "$ARCH.tmp/results"
"$PY" scripts/206_summarize_relational_slot_stage_a.py --protocol "$PROTOCOL" --predictions "$ARCH.tmp/results/predictions.jsonl" --run-metrics "$ARCH.tmp/results/run_metrics.json" --output "$ARCH.tmp/results/summary.json" --markdown "$ARCH.tmp/results/results.md"
sha256sum -c "$ARCH.tmp/frozen.sha256" > "$ARCH.tmp/verify.txt"
touch "$ARCH.tmp/COMPLETE";(cd "$ARCH.tmp";find . -type f ! -name checksums.sha256 -print0|sort -z|xargs -0 sha256sum > checksums.sha256)
mv "$ARCH.tmp" "$ARCH";echo RELATIONAL_SLOT_STAGE_A_DONE
