"""Apply the frozen Stage A relational-slot gate."""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from wmagentattack.multisource_suitability import file_sha256
from wmagentattack.multisource_suitability_experiment import paired_bootstrap, exact_sign_test

def read(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
def task_map(rows,metric):
    values=defaultdict(list)
    for row in rows:values[row["task_name"]].append(float(row[metric]))
    return {key:float(np.mean(value)) for key,value in values.items()}
def effect(left,right,metric,higher=False):
    by_seed={}
    for seed in (7,17,29):
        a=task_map([row for row in left if int(row["training_seed"])==seed],metric);b=task_map([row for row in right if int(row["training_seed"])==seed],metric)
        by_seed[seed]={task:(b[task]-a[task] if higher else a[task]-b[task]) for task in a}
    paired={task:float(np.mean([by_seed[seed][task] for seed in by_seed])) for task in by_seed[7]}
    seeds={str(seed):float(np.mean(list(values.values()))) for seed,values in by_seed.items()}
    return {"mean":float(np.mean(list(seeds.values()))),"seeds":seeds,"tasks":paired,"positive":sum(value>0 for value in paired.values())/len(paired),"bootstrap":paired_bootstrap(list(paired.values()),draws=10000,seed=818),"sign":exact_sign_test(list(paired.values()))}
def macro(rows,metric):
    by=defaultdict(list)
    for row in rows:by[(row["training_seed"],row["task_name"])].append(float(row[metric]))
    return float(np.mean([np.mean(value) for value in by.values()]))
def main():
    p=argparse.ArgumentParser();p.add_argument("--protocol",type=Path,required=True);p.add_argument("--predictions",type=Path,required=True);p.add_argument("--run-metrics",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--markdown",type=Path,required=True);args=p.parse_args()
    protocol=json.loads(args.protocol.read_text());rows=read(args.predictions);metrics=json.loads(args.run_metrics.read_text());external=read(protocol["external_v6_control"]["predictions"])
    if file_sha256(Path(protocol["external_v6_control"]["predictions"]))!=protocol["external_v6_control"]["sha256"]:raise ValueError("external v6 hash mismatch")
    baseline=[row for row in rows if row["arm"]=="v6_replication"];candidate=[row for row in rows if row["arm"]=="relational_slot_stage_a"]
    external_v6=[row for row in external if row["arm"]=="structured_residual_v6"]
    h1_base=[r for r in baseline if r["horizon"]==1];h1_candidate=[r for r in candidate if r["horizon"]==1];h1_external=[r for r in external_v6 if r["horizon"]==1]
    multi_base=[r for r in baseline if r["horizon"]>=2];multi_candidate=[r for r in candidate if r["horizon"]>=2]
    joint_base=[r for r in multi_base if r["joint_trainable"]];joint_candidate=[r for r in multi_candidate if r["joint_trainable"]]
    effects={
        "h1_nll":effect(h1_base,h1_candidate,"action_nll"),"h1_accuracy":effect(h1_base,h1_candidate,"action_correct",True),
        "h2_h5_nll":effect(multi_base,multi_candidate,"action_nll"),"future_joint_ce":effect(joint_base,joint_candidate,"joint_ce"),
        "v6_replication_nll_abs":abs(macro(h1_base,"action_nll")-macro(h1_external,"action_nll")),
        "v6_replication_accuracy_abs":abs(macro(h1_base,"action_correct")-macro(h1_external,"action_correct")),
    }
    gate=protocol["stage_a"]["gate"]
    checks={
        "h1_nll_noninferiority":effects["h1_nll"]["mean"]>=-gate["maximum_h1_nll_degradation_vs_v6"],
        "h1_accuracy_noninferiority":effects["h1_accuracy"]["mean"]>=-gate["maximum_h1_accuracy_degradation_vs_v6"],
        "h2_h5_gain":effects["h2_h5_nll"]["mean"]>=gate["minimum_h2_h5_nll_gain_vs_v6"],
        "h2_h5_task_breadth":effects["h2_h5_nll"]["positive"]>=gate["minimum_h2_h5_positive_task_fraction"],
        "h2_h5_seed_replication":sum(value>=gate["minimum_h2_h5_nll_gain_vs_v6"] for value in effects["h2_h5_nll"]["seeds"].values())>=gate["minimum_threshold_positive_seeds"],
        "future_joint_noninferiority":effects["future_joint_ce"]["mean"]>=-gate["maximum_future_joint_ce_degradation"],
        "v6_replication":max(effects["v6_replication_nll_abs"],effects["v6_replication_accuracy_abs"])<=gate["maximum_v6_replication_metric_error"],
        "zero_raw_values":not metrics["slot_audit"]["raw_values_encoded"],
        "all_legal":all(row["legal_prediction"]==1 for row in rows),
        "complete_budget":metrics["teacher_fits"]==metrics["v6_residual_fits"]==metrics["slot_residual_fits"]==15 and metrics["runtime_failures"]==0,
    }
    decision="GO_RELATIONAL_SLOT_STAGE_A" if all(checks.values()) else "NO_GO_RELATIONAL_SLOT_STAGE_A"
    summary={"protocol_id":protocol["protocol_id"],"decision":decision,"checks":checks,"effects":effects,"absolute":{"v6_h1_nll":macro(h1_base,"action_nll"),"slot_h1_nll":macro(h1_candidate,"action_nll"),"v6_h1_accuracy":macro(h1_base,"action_correct"),"slot_h1_accuracy":macro(h1_candidate,"action_correct"),"v6_h2_h5_nll":macro(multi_base,"action_nll"),"slot_h2_h5_nll":macro(multi_candidate,"action_nll"),"v6_future_joint_ce":macro(joint_base,"joint_ce"),"slot_future_joint_ce":macro(joint_candidate,"joint_ce")},"predictions_sha256":file_sha256(args.predictions),"run_metrics_sha256":file_sha256(args.run_metrics)}
    args.output.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    args.markdown.write_text("# Relational slot Stage A\n\nDecision: `"+decision+"`\n\n"+"\n".join(f"- {key}: {'PASS' if value else 'FAIL'}" for key,value in checks.items())+"\n")
if __name__=="__main__":main()
