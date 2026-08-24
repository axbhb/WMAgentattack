#!/usr/bin/env bash
#SBATCH --job-name=wma-relation-v24-smoke
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/share/guozhix/WMagentattack-relational-support-aug24/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-relational-support-aug24/logs/%x-%j.err

set -euo pipefail
WT=/share/guozhix/WMagentattack-relational-support-aug24
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
BASE=configs/0824_relation_factorized_distribution_v24_protocol.json
DATA=/share/guozhix/wmagentattack/0821/hard_label_confirmation_v21/preflight_v1/a/dataset.json
CACHE=/share/guozhix/wmagentattack/0824/relation_factorized_distribution_v24/preflight_v1/a/relation_cache.npz
ROOT=/share/guozhix/wmagentattack/0824/relation_factorized_distribution_v24/preflight_v1/smoke_v1
cd "$WT"
test ! -e "$ROOT/COMPLETE"
mkdir -p "$ROOT"
"$PY" - "$BASE" "$ROOT/smoke_protocol.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding="utf-8"))
for key in ("inner_epochs","outer_epochs","fixed_v21_epochs"):
    d["model_comparison"][key]=1
json.dump(d,open(sys.argv[2],"w",encoding="utf-8"),indent=2,sort_keys=True)
PY
PYTHONPATH=src "$PY" scripts/259_train_relation_factorized_hybrid_v24.py --protocol "$ROOT/smoke_protocol.json" --dataset "$DATA" --semantic-cache "$CACHE" --output-dir "$ROOT/results" > "$ROOT/stdout.json"
"$PY" - "$ROOT/results/run_metrics.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding="utf-8"))
assert d["completed_model_fits"]==45
assert d["completed_metric_rows"]==45
assert d["runtime_failures"]==0
print(json.dumps({"fits":45,"rows":45,"runtime_failures":0}))
PY
date -Is > "$ROOT/COMPLETE"
