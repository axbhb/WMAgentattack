"""Exactly recombine frozen v12 action and v6 joint-outcome predictions."""

from __future__ import annotations

import argparse,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from wmagentattack.multisource_suitability import file_sha256
from wmagentattack.multisource_suitability_experiment import exact_sign_test,paired_bootstrap

KEYS=("fold","training_seed","horizon","event_id")
ACTION_FIELDS=("action_nll","action_correct","legal_prediction")
JOINT_FIELDS=("joint_trainable","joint_ce")


def _read(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
def _key(row):return tuple(row[name] for name in KEYS)
def _task_means(rows,metric):
    values=defaultdict(list)
    for row in rows:
        if row.get(metric) is not None:values[row["task_name"]].append(float(row[metric]))
    return {task:float(np.mean(entries)) for task,entries in values.items()}
def _effect(left,right,metric,*,higher=False,seed=81300):
    per_seed={}
    for training_seed in (7,17,29):
        a=_task_means([row for row in left if int(row["training_seed"])==training_seed],metric);b=_task_means([row for row in right if int(row["training_seed"])==training_seed],metric)
        if set(a)!=set(b):raise ValueError("task mismatch")
        per_seed[training_seed]={task:(b[task]-a[task] if higher else a[task]-b[task]) for task in a}
    tasks=set(per_seed[7]);paired={task:float(np.mean([per_seed[s][task] for s in per_seed])) for task in tasks};seeds={str(s):float(np.mean(list(v.values()))) for s,v in per_seed.items()}
    return {"mean":float(np.mean(list(seeds.values()))),"seeds":seeds,"tasks":paired,"positive_task_fraction":float(np.mean([v>0 for v in paired.values()])),"paired_bootstrap":paired_bootstrap(list(paired.values()),draws=10000,seed=seed),"exact_sign_test":exact_sign_test(list(paired.values()))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--protocol",type=Path,required=True);parser.add_argument("--output-dir",type=Path,required=True);args=parser.parse_args()
    protocol=json.loads(args.protocol.read_text());sources=protocol["frozen_sources"]
    if protocol["status"]!="preregistered_modular_recombination_before_build":raise ValueError("v13 protocol not frozen")
    for path_key,hash_key in (("v12_predictions","v12_predictions_sha256"),("v12_gate","v12_gate_sha256"),("v6_predictions","v6_predictions_sha256")):
        if file_sha256(Path(sources[path_key]))!=sources[hash_key]:raise ValueError(f"source hash mismatch {path_key}")
    v12_all=_read(sources["v12_predictions"]);v12=[row for row in v12_all if row["arm"]==protocol["stage_m1"]["action_source_arm"]]
    v6_all=_read(sources["v6_predictions"]);v6=[row for row in v6_all if row["arm"]==protocol["stage_m1"]["outcome_source_arm"]]
    left={_key(row):row for row in v12};right={_key(row):row for row in v6}
    if len(left)!=len(v12) or len(right)!=len(v6) or set(left)!=set(right):raise ValueError("non-bijective pairing")
    composite=[]
    for key in sorted(left):
        action=left[key];outcome=right[key];row=dict(action);row["arm"]=protocol["stage_m1"]["composite_arm"]
        for name in JOINT_FIELDS:row[name]=outcome[name]
        composite.append(row)
    args.output_dir.mkdir(parents=True,exist_ok=True);pred=args.output_dir/"predictions.jsonl"
    pred.write_text("".join(json.dumps(row,sort_keys=True)+"\n" for row in composite))
    action_exact=all(all(row[name]==left[_key(row)][name] for name in ACTION_FIELDS) for row in composite)
    joint_exact=all(all(row[name]==right[_key(row)][name] for name in JOINT_FIELDS) for row in composite)
    effects={
        "h1_nll_vs_v6":_effect([r for r in v6 if r["horizon"]==1],[r for r in composite if r["horizon"]==1],"action_nll",seed=81331),
        "h1_accuracy_vs_v6":_effect([r for r in v6 if r["horizon"]==1],[r for r in composite if r["horizon"]==1],"action_correct",higher=True,seed=81332),
        "h2_h5_nll_vs_v6":_effect([r for r in v6 if r["horizon"]>=2],[r for r in composite if r["horizon"]>=2],"action_nll",seed=81333),
        "future_joint_ce_vs_v6":_effect([r for r in v6 if r["horizon"]>=2 and r["joint_trainable"]],[r for r in composite if r["horizon"]>=2 and r["joint_trainable"]],"joint_ce",seed=81334),
    }
    v12_gate=json.loads(Path(sources["v12_gate"]).read_text());capacity_gain=v12_gate["effects"]["h2_h5_nll_vs_zero_graph"]["mean"]
    gate=protocol["stage_m1"]["gate"]
    checks={"expected_rows":len(composite)==protocol["stage_m1"]["expected_rows"],"exact_action_preservation":action_exact,"exact_joint_preservation":joint_exact,"h1_nll_noninferiority":effects["h1_nll_vs_v6"]["mean"]>=-gate["maximum_h1_nll_degradation_vs_v6"],"h1_accuracy_noninferiority":effects["h1_accuracy_vs_v6"]["mean"]>=-gate["maximum_h1_accuracy_degradation_vs_v6"],"h2_h5_gain_vs_v6":effects["h2_h5_nll_vs_v6"]["mean"]>=gate["minimum_h2_h5_nll_gain_vs_v6"],"h2_h5_gain_vs_zero_graph":capacity_gain>=gate["minimum_h2_h5_nll_gain_vs_v12_zero_graph"],"task_breadth":effects["h2_h5_nll_vs_v6"]["positive_task_fraction"]>=gate["minimum_positive_task_fraction"],"seed_replication":sum(v>0 for v in effects["h2_h5_nll_vs_v6"]["seeds"].values())>=gate["minimum_positive_seeds"],"future_joint_exact":abs(effects["future_joint_ce_vs_v6"]["mean"])<=gate["maximum_absolute_future_joint_ce_difference_vs_v6"],"all_legal":all(row["legal_prediction"]==1 for row in composite)}
    decision="GO_MODULAR_EVENT_WORLD_MODEL_V13" if all(checks.values()) else "NO_GO_MODULAR_EVENT_WORLD_MODEL_V13"
    summary={"protocol_id":protocol["protocol_id"],"decision":decision,"gate_checks":checks,"effects":effects,"carried_v12_h2_h5_gain_vs_zero_graph":capacity_gain,"rows":len(composite),"predictions_sha256":file_sha256(pred),"training_runs":0}
    (args.output_dir/"modular_gate.json").write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n")
    (args.output_dir/"results.md").write_text("# Modular event world model v13\n\nDecision: `"+decision+"`\n\n"+"\n".join(f"- {k}: {'PASS' if v else 'FAIL'}" for k,v in checks.items())+"\n")
    if not all(checks.values()):raise SystemExit(2)


if __name__=="__main__":main()
