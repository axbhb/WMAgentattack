#!/usr/bin/env bash
#SBATCH --job-name=wm-relroute-e1
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/0818_relational_router_e1_%j.out
#SBATCH --error=logs/0818_relational_router_e1_%j.err
set -euo pipefail
ROOT=/share/guozhix/WMagentattack-relational-router-aug18
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
ARCH=/share/guozhix/wmagentattack/0818/relational_router_uncertainty_v10/stage_e1/formal_v1
DATA=/share/guozhix/wmagentattack/0817/structured_joint_outcome_v5/preflight_v1/a/dataset.json
AUDIT=/share/guozhix/wmagentattack/0817/structured_joint_outcome_v5/preflight_v1/a/audit.json
PROTOCOL=$ROOT/configs/0818_relational_router_uncertainty_protocol.json
cd "$ROOT"; export PYTHONPATH="$ROOT/src"; mkdir -p logs
if [[ -e "$ARCH/COMPLETE" ]]; then exit 3; fi
rm -rf "$ARCH.tmp"; mkdir -p "$ARCH.tmp/results" "$ARCH.tmp/source_snapshot"
printf 'job_id=%s\ngit_commit=%s\n' "$SLURM_JOB_ID" "$(git rev-parse HEAD)" > "$ARCH.tmp/runtime.txt"
cp "$PROTOCOL" "$ARCH.tmp/protocol.json"
SOURCES=(
  src/wmagentattack/relational_router_residual.py
  src/wmagentattack/relational_slot_latent.py
  src/wmagentattack/domain_expert_latent.py
  src/wmagentattack/structured_residual_dynamics.py
  src/wmagentattack/joint_outcome_auxiliary.py
  scripts/201_train_structured_joint_outcome_v5.py
  scripts/203_train_structured_residual_v6.py
  scripts/211_summarize_interface_affordance_c1.py
  scripts/214_train_domain_expert_d1.py
  scripts/216_train_relational_router_e1.py
  scripts/217_summarize_relational_router_e1.py
)
cp "${SOURCES[@]}" "$ARCH.tmp/source_snapshot/"
sha256sum "$PROTOCOL" "$DATA" "$AUDIT" "${SOURCES[@]}" > "$ARCH.tmp/frozen.sha256"
"$PY" -m pytest -q tests/test_relational_router_residual.py tests/test_structured_residual_dynamics.py tests/test_joint_outcome_auxiliary.py > "$ARCH.tmp/tests.txt"
"$PY" scripts/216_train_relational_router_e1.py --protocol "$PROTOCOL" --dataset "$DATA" --audit "$AUDIT" --output-dir "$ARCH.tmp/results"
"$PY" scripts/217_summarize_relational_router_e1.py --protocol "$PROTOCOL" --predictions "$ARCH.tmp/results/predictions.jsonl" --run-metrics "$ARCH.tmp/results/run_metrics.json" --output "$ARCH.tmp/results/summary.json" --markdown "$ARCH.tmp/results/results.md"
sha256sum -c "$ARCH.tmp/frozen.sha256" > "$ARCH.tmp/verify.txt"
touch "$ARCH.tmp/COMPLETE"; (cd "$ARCH.tmp"; find . -type f ! -name checksums.sha256 -print0 | sort -z | xargs -0 sha256sum > checksums.sha256)
mv "$ARCH.tmp" "$ARCH"; echo RELATIONAL_ROUTER_E1_DONE
