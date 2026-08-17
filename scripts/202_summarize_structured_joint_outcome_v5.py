"""Apply the frozen v5 task-level auxiliary-loss gate."""

from __future__ import annotations

import argparse,json,sys
from collections import defaultdict
from pathlib import Path
from typing import Any,Mapping,Sequence
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from wmagentattack.multisource_suitability import file_sha256
from wmagentattack.multisource_suitability_experiment import exact_sign_test,paired_bootstrap


def read(path): return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
def write(path,value): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")


def select(rows,arm,seed): return [r for r in rows if r["arm"]==arm and r["training_seed"]==seed]


def task_metric(rows,metric,tail=False):
    d=defaultdict(list)
    for r in rows:
        if tail and not r["has_next_action"]: continue
        if r.get(metric) is not None: d[r["task_name"]].append(float(r[metric]))
    return {k:float(np.mean(v)) for k,v in sorted(d.items())}


def joint_task_metric(rows,metric):
    traj=defaultdict(list); group_task={}; group=defaultdict(list); task=defaultdict(list)
    for r in rows:
        if not r["joint_trainable"]: continue
        traj[(r["joint_group_id"],r["trajectory_id"],r["task_name"])].append(float(r[metric]))
    for (g,t,task_name),v in traj.items(): group[g].append(float(np.mean(v))); group_task[g]=task_name
    for g,v in group.items(): task[group_task[g]].append(float(np.mean(v)))
    return {k:float(np.mean(v)) for k,v in sorted(task.items())}


def effect(rows,seeds,metric,higher=False,joint=False):
    by_seed={}
    for seed in seeds:
        a=select(rows,"structured_baseline",seed); b=select(rows,"structured_joint_aux",seed)
        fn=joint_task_metric if joint else task_metric
        if joint: left,right=fn(a,metric),fn(b,metric)
        else: left,right=fn(a,metric,tail=metric.startswith("action_")),fn(b,metric,tail=metric.startswith("action_"))
        by_seed[seed]={t:(right[t]-left[t] if higher else left[t]-right[t]) for t in left}
    tasks=set(next(iter(by_seed.values())))
    paired={t:float(np.mean([by_seed[s][t] for s in seeds])) for t in sorted(tasks)}
    gains={str(s):float(np.mean(list(by_seed[s].values()))) for s in seeds}
    return {"mean_gain":float(np.mean(list(gains.values()))),"gain_by_seed":gains,"paired_task_gains":paired,"positive_task_fraction":sum(v>0 for v in paired.values())/len(paired),"bootstrap":paired_bootstrap(list(paired.values()),draws=10000,seed=8172500+len(metric)),"sign_test":exact_sign_test(list(paired.values()))}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--protocol",type=Path,required=True); p.add_argument("--dataset",type=Path,required=True); p.add_argument("--audit",type=Path,required=True); p.add_argument("--predictions",type=Path,required=True); p.add_argument("--run-metrics",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--markdown",type=Path,required=True); a=p.parse_args()
    protocol=json.loads(a.protocol.read_text()); metrics=json.loads(a.run_metrics.read_text()); rows=read(a.predictions); seeds=protocol["training"]["training_seeds"]
    if metrics["neural_training_runs"]!=protocol["fixed_budget"]["neural_training_runs"] or metrics["predictions_sha256"]!=file_sha256(a.predictions): raise ValueError("incomplete or changed predictions")
    effects={
      "action_nll":effect(rows,seeds,"action_nll"),
      "action_accuracy":effect(rows,seeds,"action_correct",higher=True),
      "observable_outcome_bce":effect(rows,seeds,"outcome_bce"),
      "joint_cross_entropy":effect(rows,seeds,"joint_cross_entropy",joint=True),
      "joint_brier":effect(rows,seeds,"joint_brier",joint=True),
    }
    gate=protocol["acceptance_gate"]
    checks={
      "action_nll_gain":effects["action_nll"]["mean_gain"]>=gate["minimum_action_nll_gain"],
      "action_accuracy_gain":effects["action_accuracy"]["mean_gain"]>=gate["minimum_action_accuracy_gain"],
      "action_seed_replication":sum(v>=gate["minimum_action_nll_gain"] for v in effects["action_nll"]["gain_by_seed"].values())>=gate["minimum_positive_seeds"],
      "action_positive_task_fraction":effects["action_nll"]["positive_task_fraction"]>=gate["minimum_positive_task_fraction"],
      "outcome_bce_noninferiority":effects["observable_outcome_bce"]["mean_gain"]>=-gate["maximum_outcome_bce_degradation"],
      "joint_ce_gain_over_train_prior":effects["joint_cross_entropy"]["mean_gain"]>=gate["minimum_joint_ce_gain"],
      "joint_brier_gain_over_train_prior":effects["joint_brier"]["mean_gain"]>=gate["minimum_joint_brier_gain"],
      "joint_seed_replication":sum(v>=gate["minimum_joint_ce_gain"] for v in effects["joint_cross_entropy"]["gain_by_seed"].values())>=gate["minimum_positive_seeds"],
      "joint_positive_task_fraction":effects["joint_cross_entropy"]["positive_task_fraction"]>=gate["minimum_positive_task_fraction"],
      "all_predictions_legal":all(r["legal_prediction"]==1 for r in rows),
      "complete_budget":metrics["neural_training_runs"]==protocol["fixed_budget"]["neural_training_runs"],
    }
    passed=all(checks.values()); decision="GO_RETAIN_STRUCTURED_JOINT_AUXILIARY" if passed else "NO_GO_STRUCTURED_JOINT_AUXILIARY_DOES_NOT_IMPROVE_DYNAMICS"
    summary={"protocol_id":protocol["protocol_id"],"decision":decision,"gate_passed":passed,"gate_checks":checks,"effects":effects,"run":{"job_id":int(metrics.get("job_id",0)),"neural_training_runs":metrics["neural_training_runs"],"prediction_rows":len(rows),"predictions_sha256":file_sha256(a.predictions),"run_metrics_sha256":file_sha256(a.run_metrics),"runtime_failures":0},"dataset":{"sha256":file_sha256(a.dataset),"audit_sha256":file_sha256(a.audit)}}
    write(a.output,summary)
    lines=["# Structured Markov joint-outcome auxiliary v5", "",f"Decision: `{decision}`","","| metric | task-macro gain | positive tasks |","|---|---:|---:|"]
    for k,v in effects.items(): lines.append(f"| {k} | {v['mean_gain']:+.6f} | {v['positive_task_fraction']:.1%} |")
    lines += ["","## Frozen gate",""]+[f"- {k}: **{'PASS' if v else 'FAIL'}**" for k,v in checks.items()]
    a.markdown.parent.mkdir(parents=True,exist_ok=True); a.markdown.write_text("\n".join(lines)+"\n",encoding="utf-8")

if __name__=="__main__": main()
