"""Apply the frozen v12 oracle-sufficiency gate."""

from __future__ import annotations

import argparse,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from wmagentattack.multisource_suitability import file_sha256
from wmagentattack.multisource_suitability_experiment import exact_sign_test,paired_bootstrap


def _read(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _task_means(rows,metric):
    values=defaultdict(list)
    for row in rows:
        if row.get(metric) is not None:values[row["task_name"]].append(float(row[metric]))
    return {task:float(np.mean(entries)) for task,entries in values.items()}


def _effect(left,right,metric,*,higher=False,seed=81200):
    per_seed={}
    for training_seed in (7,17,29):
        a=_task_means([row for row in left if int(row["training_seed"])==training_seed],metric)
        b=_task_means([row for row in right if int(row["training_seed"])==training_seed],metric)
        if set(a)!=set(b):raise ValueError(f"task mismatch {metric} {training_seed}")
        per_seed[training_seed]={task:(b[task]-a[task] if higher else a[task]-b[task]) for task in a}
    tasks=set(per_seed[7]);paired={task:float(np.mean([per_seed[s][task] for s in per_seed])) for task in tasks}
    seeds={str(s):float(np.mean(list(values.values()))) for s,values in per_seed.items()}
    return {"mean":float(np.mean(list(seeds.values()))),"seeds":seeds,"tasks":paired,"positive_task_fraction":float(np.mean([value>0 for value in paired.values()])),"paired_bootstrap":paired_bootstrap(list(paired.values()),draws=10000,seed=seed),"exact_sign_test":exact_sign_test(list(paired.values()))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--predictions",type=Path,required=True);parser.add_argument("--run-metrics",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--markdown",type=Path,required=True);args=parser.parse_args()
    protocol=json.loads(args.protocol.read_text());current=_read(args.predictions);metrics=json.loads(args.run_metrics.read_text())
    external_path=Path(protocol["external_v6_control"]["predictions"])
    if file_sha256(external_path)!=protocol["external_v6_control"]["sha256"]:raise ValueError("v6 hash mismatch")
    external=_read(external_path);v6=[row for row in external if row["arm"]=="structured_residual_v6"]
    zero=[row for row in current if row["arm"]=="zero_graph_capacity_control_v12"]
    oracle=[row for row in current if row["arm"]=="true_event_graph_oracle_v12"]
    if not len(v6)==len(zero)==len(oracle):raise ValueError("paired row mismatch")
    effects={
        "h1_nll_vs_v6":_effect([r for r in v6 if r["horizon"]==1],[r for r in oracle if r["horizon"]==1],"action_nll",seed=81231),
        "h1_accuracy_vs_v6":_effect([r for r in v6 if r["horizon"]==1],[r for r in oracle if r["horizon"]==1],"action_correct",higher=True,seed=81232),
        "h2_h5_nll_vs_v6":_effect([r for r in v6 if r["horizon"]>=2],[r for r in oracle if r["horizon"]>=2],"action_nll",seed=81233),
        "h2_h5_nll_vs_zero_graph":_effect([r for r in zero if r["horizon"]>=2],[r for r in oracle if r["horizon"]>=2],"action_nll",seed=81234),
        "future_joint_ce_vs_v6":_effect([r for r in v6 if r["horizon"]>=2 and r["joint_trainable"]],[r for r in oracle if r["horizon"]>=2 and r["joint_trainable"]],"joint_ce",seed=81235),
    }
    gate=protocol["oracle_sufficiency_stage"]["gate"]
    checks={
        "complete_budget":metrics["training_units"]==30,
        "runtime_clean":metrics["runtime_failures"]==0,
        "parameter_match":bool(metrics["parameter_match"]),
        "paired_rows_complete":len(oracle)==len(v6),
        "h1_nll_noninferiority":effects["h1_nll_vs_v6"]["mean"]>=-gate["maximum_h1_nll_degradation_vs_v6"],
        "h1_accuracy_noninferiority":effects["h1_accuracy_vs_v6"]["mean"]>=-gate["maximum_h1_accuracy_degradation_vs_v6"],
        "h2_h5_gain_vs_v6":effects["h2_h5_nll_vs_v6"]["mean"]>=gate["minimum_h2_h5_nll_gain_vs_v6"],
        "h2_h5_gain_vs_zero_graph":effects["h2_h5_nll_vs_zero_graph"]["mean"]>=gate["minimum_h2_h5_nll_gain_vs_zero_graph_control"],
        "task_breadth":effects["h2_h5_nll_vs_v6"]["positive_task_fraction"]>=gate["minimum_positive_task_fraction"],
        "seed_replication":sum(value>0 for value in effects["h2_h5_nll_vs_v6"]["seeds"].values())>=gate["minimum_positive_seeds"],
        "future_joint_gain":effects["future_joint_ce_vs_v6"]["mean"]>=gate["minimum_future_joint_ce_gain_vs_v6"],
        "all_legal":all(row["legal_prediction"]==1 for row in current),
    }
    decision="GO_EVENT_GRAPH_ORACLE_SUFFICIENCY_V12" if all(checks.values()) else "NO_GO_EVENT_GRAPH_ORACLE_SUFFICIENCY_V12"
    diagnosis="True action-event graphs contain sufficient transferable dynamics information; a learned graph predictor is authorized." if decision.startswith("GO") else "Even true action-event graphs do not clear the frozen sufficiency gate; a learned graph predictor is not authorized."
    summary={"protocol_id":protocol["protocol_id"],"decision":decision,"diagnosis":diagnosis,"gate_checks":checks,"effects":effects,"counts":{"v6":len(v6),"zero":len(zero),"oracle":len(oracle)},"predictions_sha256":file_sha256(args.predictions),"run_metrics_sha256":file_sha256(args.run_metrics),"oracle_is_diagnostic_only":True}
    args.output.write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n")
    lines=["# Action-event graph v12 oracle sufficiency","",f"Decision: `{decision}`","",diagnosis,"","## Frozen gate",""]
    lines.extend(f"- {name}: {'PASS' if passed else 'FAIL'}" for name,passed in checks.items())
    lines.extend(["","## Paired effects",""])
    lines.extend(f"- {name}: {value['mean']:.6f}" for name,value in effects.items())
    lines.extend(["","True future graphs are diagnostic only and cannot be used as deployed model inputs.",""])
    args.markdown.write_text("\n".join(lines))


if __name__=="__main__":main()
