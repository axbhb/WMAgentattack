#!/usr/bin/env bash
set -euo pipefail

REPO=${WMAGENTATTACK_REPO:-/home/pth/projects/WMagentattack-v45}
PY=${WMAGENTATTACK_PYTHON:-/home/pth/venvs/wmagentattack_v45/bin/python}
ARCHIVE=${WMAGENTATTACK_ARCHIVE:-/home/pth/outputs/wmagentattack/v45_formal_v1}
DATASET=${WMAGENTATTACK_DATASET:-/home/pth/data/wmagentattack/v45/dataset.json}
E5=${WMAGENTATTACK_E5:-/home/pth/models/e5-base-v2}
V5=${WMAGENTATTACK_V5:-/home/pth/data/wmagentattack/v45/baselines/v5_predictions.jsonl}
V22=${WMAGENTATTACK_V22:-/home/pth/data/wmagentattack/v45/baselines/v22_predictions.jsonl}
PROTOCOL=${REPO}/configs/0902_large_hybrid_world_model_v45_protocol.json
CACHE=${ARCHIVE}/semantic_cache/large_semantic_cache_v45.npz
CACHE_META=${ARCHIVE}/semantic_cache/large_semantic_cache_v45_metadata.json
RUNS=${ARCHIVE}/runs
LOGS=${ARCHIVE}/logs
LOCK=${ARCHIVE}/.coordinator.lock
SEEDS=(7 17 29)

mkdir -p "${ARCHIVE}" "${ARCHIVE}/semantic_cache" "${RUNS}" "${LOGS}"
if ! mkdir "${LOCK}" 2>/dev/null; then
  echo "A v45 coordinator lock already exists: ${LOCK}" >&2
  exit 3
fi

finish() {
  status=$?
  if [[ ${status} -ne 0 ]]; then
    printf '%s status=%s\n' "$(date -Is)" "${status}" > "${ARCHIVE}/FAILED"
  fi
  rmdir "${LOCK}" 2>/dev/null || true
  exit "${status}"
}
trap finish EXIT

cd "${REPO}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

"${PY}" - "${DATASET}" "${E5}" "${V5}" "${V22}" <<'PY'
import json
import sys
from pathlib import Path
import torch

dataset, model, v5, v22 = map(Path, sys.argv[1:])
for path in (dataset, model, v5, v22):
    if not path.exists():
        raise SystemExit(f"missing formal input: {path}")
payload = json.loads(dataset.read_text(encoding="utf-8"))
if len(payload["events"]) != 6763 or len(payload["candidate_catalog"]) != 31:
    raise SystemExit("formal dataset cardinality mismatch")
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise SystemExit("two CUDA devices are required")
print(json.dumps({
    "events": len(payload["events"]),
    "candidates": len(payload["candidate_catalog"]),
    "cuda_devices": torch.cuda.device_count(),
    "device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}, sort_keys=True))
PY

if [[ ! -f "${CACHE}" || ! -f "${CACHE_META}" ]]; then
  CUDA_VISIBLE_DEVICES=0 "${PY}" scripts/297_build_large_semantic_cache_v45.py \
    --dataset "${DATASET}" \
    --model "${E5}" \
    --output "${CACHE}" \
    --metadata "${CACHE_META}" \
    --batch-size 64 \
    --max-length 256 \
    --device cuda \
    > "${LOGS}/semantic_cache.out" \
    2> "${LOGS}/semantic_cache.err"
fi

"${PY}" - "${CACHE}" "${CACHE_META}" <<'PY'
import json
import sys
import numpy as np

cache = np.load(sys.argv[1])
meta = json.load(open(sys.argv[2], encoding="utf-8"))
if list(cache["field_embeddings"].shape) != [6763, 5, 768]:
    raise SystemExit("formal field cache shape mismatch")
if list(cache["candidate_embeddings"].shape) != [31, 768]:
    raise SystemExit("formal candidate cache shape mismatch")
if not np.isfinite(cache["field_embeddings"]).all() or not np.isfinite(cache["candidate_embeddings"]).all():
    raise SystemExit("non-finite formal semantic cache")
if meta.get("outcome_fields_encoded") != 0 or meta.get("task_identifiers_encoded") != 0:
    raise SystemExit("forbidden fields encoded in formal semantic cache")
print(json.dumps({"semantic_cache": "ready", "events": meta["events"], "candidates": meta["candidates"]}))
PY

run_fit() {
  local gpu=$1
  local index=$2
  local fold=$((index / 3))
  local seed=${SEEDS[$((index % 3))]}
  local out="${RUNS}/fold${fold}_seed${seed}"
  if [[ -f "${out}/metrics.json" && -f "${out}/predictions.jsonl" && -f "${out}/checkpoint.pt" ]]; then
    echo "Skipping already complete fold=${fold} seed=${seed}"
    return 0
  fi
  if [[ -e "${out}" ]]; then
    echo "Refusing partial or mismatched formal output: ${out}" >&2
    return 4
  fi
  echo "Starting fold=${fold} seed=${seed} physical_gpu=${gpu} at $(date -Is)"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/298_train_large_hybrid_world_model_v45.py \
    --protocol "${PROTOCOL}" \
    --dataset "${DATASET}" \
    --semantic-cache "${CACHE}" \
    --output-dir "${RUNS}" \
    --fold "${fold}" \
    --seed "${seed}" \
    --device cuda \
    > "${LOGS}/fold${fold}_seed${seed}.out" \
    2> "${LOGS}/fold${fold}_seed${seed}.err"
  echo "Completed fold=${fold} seed=${seed} physical_gpu=${gpu} at $(date -Is)"
}

worker() {
  local gpu=$1
  shift
  local index
  for index in "$@"; do
    run_fit "${gpu}" "${index}"
  done
}

worker 0 0 2 4 6 8 10 12 14 > "${LOGS}/worker_gpu0.out" 2> "${LOGS}/worker_gpu0.err" &
pid0=$!
worker 1 1 3 5 7 9 11 13 > "${LOGS}/worker_gpu1.out" 2> "${LOGS}/worker_gpu1.err" &
pid1=$!
set +e
wait "${pid0}"
status0=$?
wait "${pid1}"
status1=$?
set -e
if [[ ${status0} -ne 0 || ${status1} -ne 0 ]]; then
  echo "Formal worker failure: gpu0=${status0} gpu1=${status1}" >&2
  exit 5
fi

"${PY}" scripts/299_gate_large_hybrid_world_model_v45.py \
  --protocol "${PROTOCOL}" \
  --dataset "${DATASET}" \
  --runs-root "${RUNS}" \
  --v5-predictions "${V5}" \
  --v22-predictions "${V22}" \
  --output "${ARCHIVE}/large_hybrid_world_model_v45_summary.json" \
  --markdown "${ARCHIVE}/large_hybrid_world_model_v45_results.md" \
  > "${LOGS}/gate.out" \
  2> "${LOGS}/gate.err"

rm -f "${ARCHIVE}/FAILED"
printf '%s\n' "$(date -Is)" > "${ARCHIVE}/COMPLETE"
