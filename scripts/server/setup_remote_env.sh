#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/guozhix/WMagentattack}"
CONDA_BIN="${CONDA_BIN:-/share/guozhix/software/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-wmagentattack}"
AGENTDOJO_REPO="${AGENTDOJO_REPO:-https://github.com/ethz-spylab/agentdojo.git}"
AGENTDOJO_COMMIT="${AGENTDOJO_COMMIT:-089ed468cf3ed0322acc66b0211f26d9d90dbf60}"

cd "$PROJECT_ROOT"

if [ ! -d external/agentdojo/.git ]; then
  mkdir -p external
  git clone "$AGENTDOJO_REPO" external/agentdojo
fi
git -C external/agentdojo fetch --all --tags
git -C external/agentdojo checkout "$AGENTDOJO_COMMIT"

if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  "$CONDA_BIN" create -y -n "$ENV_NAME" python=3.11 pip
fi

eval "$("$CONDA_BIN" shell.bash hook)"
conda activate "$ENV_NAME"

python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
  torch==2.10.0 transformers==5.12.1 accelerate==1.14.0 \
  bitsandbytes==0.49.2 safetensors==0.8.0 \
  pytest==9.1.1 scikit-learn==1.9.0 joblib==1.5.3 pyyaml
python -m pip install -e external/agentdojo

python -m pip check
python -m pytest -q

