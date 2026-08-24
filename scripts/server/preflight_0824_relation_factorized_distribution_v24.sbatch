#!/usr/bin/env bash
#SBATCH --job-name=wma-relation-v24-pre
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/share/guozhix/WMagentattack-relational-support-aug24/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-relational-support-aug24/logs/%x-%j.err

set -euo pipefail
WT=/share/guozhix/WMagentattack-relational-support-aug24
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
PROTOCOL=configs/0824_relation_factorized_distribution_v24_protocol.json
DATA=/share/guozhix/wmagentattack/0821/hard_label_confirmation_v21/preflight_v1/a/dataset.json
ROOT=/share/guozhix/wmagentattack/0824/relation_factorized_distribution_v24/preflight_v1
cd "$WT"
test ! -e "$ROOT/COMPLETE"
mkdir -p "$ROOT/a" "$ROOT/b"
PYTHONPATH=src "$PY" -m pytest -q tests/test_relation_factorized_semantic_world_model.py > "$ROOT/tests.txt"
for replica in a b; do
  PYTHONPATH=src "$PY" scripts/258_build_relation_factorized_cache_v24.py +    --protocol "$PROTOCOL" --dataset "$DATA" +    --output "$ROOT/$replica/relation_cache.npz" +    --audit "$ROOT/$replica/audit.json" > "$ROOT/$replica/stdout.json"
done
PYTHONPATH=src "$PY" - "$ROOT" <<'PY'
import hashlib,json,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1])
a=np.load(root/"a/relation_cache.npz")
b=np.load(root/"b/relation_cache.npz")
keys=sorted(a.files)
same={}
for key in keys:
    left,right=a[key],b[key]
    same[key]=bool(np.array_equal(left,right))
audit_a=json.loads((root/"a/audit.json").read_text())
audit_b=json.loads((root/"b/audit.json").read_text())
for audit in (audit_a,audit_b):
    audit.pop("cache_sha256",None)
payload={
    "schema_version":"wmagentattack.relation_factorized_cache_double_build.v24",
    "array_keys":keys,
    "arrays_identical":same,
    "all_arrays_identical":all(same.values()),
    "audits_identical_excluding_container_hash":audit_a==audit_b,
    "outcome_fields_consumed":audit_a["outcome_fields_consumed"],
    "real_external_endpoint_calls":audit_a["real_external_endpoint_calls"],
}
(root/"double_build_gate.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
assert payload["all_arrays_identical"]
assert payload["audits_identical_excluding_container_hash"]
assert payload["outcome_fields_consumed"]==[]
assert payload["real_external_endpoint_calls"]==0
PY
date -Is > "$ROOT/COMPLETE"
