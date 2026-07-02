#!/usr/bin/env bash
# Login-node safe bootstrap for NTU HPCC GaaS.
#
# This script only clones/pulls code, creates lightweight directories, and
# prints PBS commands.  Do not run training, inference, package builds, or long
# Python commands on the login node.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/WMagentattack}"
REPO_URL="${REPO_URL:-https://github.com/axbhb/WMAgentattack.git}"
BRANCH="${BRANCH:-main}"

echo "[gaas] Host: $(hostname)"
echo "[gaas] Project root: $PROJECT_ROOT"

if [ ! -d "$PROJECT_ROOT/.git" ]; then
  mkdir -p "$(dirname "$PROJECT_ROOT")"
  git clone --branch "$BRANCH" "$REPO_URL" "$PROJECT_ROOT"
else
  cd "$PROJECT_ROOT"
  git fetch origin "$BRANCH"
  git checkout "$BRANCH"
  git pull --ff-only origin "$BRANCH"
fi

cd "$PROJECT_ROOT"
mkdir -p logs runs artifacts data external

echo "[gaas] Queue overview:"
qstat -Q || true

echo "[gaas] Your current jobs:"
qstat -u "$USER" || true

cat <<'EOF'

[gaas] Code is deployed. Next recommended steps:

1) Create/install the conda environment on a compute node:
   qsub scripts/server/gaas_setup_env.pbs

   If GaaS requires a project code, copy the PBS file and add:
   #PBS -P <PROJECT_CODE>

2) Run a short GPU/import smoke test:
   qsub scripts/server/gaas_smoke_test.pbs

3) For this repo's conditional-preservation experiment, first make sure the
   AgentDojo data/artifacts and model cache exist on GaaS, then submit the PBS
   scripts under scripts/server/gaas_*.pbs.

EOF

