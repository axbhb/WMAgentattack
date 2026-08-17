"""Apply the frozen interface-affordance C1 gate."""
from __future__ import annotations
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from wmagentattack.multisource_suitability import file_sha256
from wmagentattack.multisource_suitability_experiment import paired_bootstrap,exact_sign_test
def read(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
def task_map(rows,metric):
 values=defaultdict(list)
 for row in rows:values[row["task_name"]].append(float(row[metric]))
 return {key:float(np.mean(value)) for key,value in values.items()}
def effect(left,right,metric,higher=False):
 by_seed={}
 for seed in (7,17,29):
  a=task_map([r for r in left if int(r["training_seed"])==seed],metric);b=task_map([r for r in right if int(r["training_seed"])==seed],metric)
  if set(a)!=set(b):raise ValueError("paired task surface mismatch")
  by_seed[seed]={task:(b[task]-a[task] if higher else a[task]-b[task]) for task in a}
 paired={task:float(np.mean([by_seed[seed][task] for seed in by_seed])) for task in by_seed[7]};seeds={str(seed):float(np.mean(list(values.values()))) for seed,values in by_seed.items()}
 return {"mean":float(np.mean(list(seeds.values()))),"seeds":seeds,"tasks":paired,"positive":sum(v>0 for v in paired.values())/len(paired),"bootstrap":paired_bootstrap(list(paired.values()),draws=10000,seed=818),"sign":exact_sign_test(list(paired.values()))}
def macro(rows,metric):
 by=defaultdict(list)
 for row in rows:by[(row["training_seed"],row["task_name"])].append(float(row[metric]))
 return float(np.mean([np.mean(v) for v in by.values()]))
def main():
 p=argparse.ArgumentParser();p.add_argument("--protocol",type=Path,required=True);p.add_argument("--predictions",type=Path,required=True);p.add_argument("--run-metrics",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--markdown",type=Path,required=True);a=p.parse_args()
 protocol=json.loads(a.protocol.read_text());rows=read(a.predictions);metrics=json.loads(a.run_metrics.read_text());external=read(protocol["external_v6_control"]["predictions"])
 if file_sha256(Path(protocol["external_v6_control"]["predictions"]))!=protocol["external_v6_control"]["sha256"]:raise ValueError("external v6 mismatch")
 baseline=[r for r in external if r["arm"]=="structured_residual_v6"];candidate=[r for r in rows if r["arm"]=="interface_affordance_c1"]
 h1b=[r for r in baseline if r["horizon"]==1];h1c=[r for r in candidate if r["horizon"]==1];mb=[r for r in baseline if r["horizon"]>=2];mc=[r for r in candidate if r["horizon"]>=2];jb=[r for r in mb if r["joint_trainable"]];jc=[r for r in mc if r["joint_trainable"]]
 effects={"h1_nll":effect(h1b,h1c,"action_nll"),"h1_accuracy":effect(h1b,h1c,"action_correct",True),"h2_h5_nll":effect(mb,mc,"action_nll"),"future_joint_ce":effect(jb,jc,"joint_ce")};g=protocol["stage_c1"]["gate"];audit=metrics["slot_audit"]
 checks={"h1_nll_noninferiority":effects["h1_nll"]["mean"]>=-g["maximum_h1_nll_degradation_vs_v6"],"h1_accuracy_noninferiority":effects["h1_accuracy"]["mean"]>=-g["maximum_h1_accuracy_degradation_vs_v6"],"h2_h5_gain":effects["h2_h5_nll"]["mean"]>=g["minimum_h2_h5_nll_gain_vs_v6"],"h2_h5_task_breadth":effects["h2_h5_nll"]["positive"]>=g["minimum_h2_h5_positive_task_fraction"],"h2_h5_seed_replication":sum(v>=g["minimum_h2_h5_nll_gain_vs_v6"] for v in effects["h2_h5_nll"]["seeds"].values())>=g["minimum_threshold_positive_seeds"],"future_joint_noninferiority":effects["future_joint_ce"]["mean"]>=-g["maximum_future_joint_ce_degradation"],"zero_raw_values":not audit["raw_values_encoded"],"interface_only":audit["interface_only_lexical_encoding"],"zero_unmatched_text":audit["unmatched_text_tokens_encoded"]==0,"zero_truncation":audit["truncated_rows"]==audit["concept_truncated_rows"]==0,"all_legal":all(r["legal_prediction"]==1 for r in candidate),"complete_budget":metrics["teacher_fits"]==metrics["affordance_residual_fits"]==15 and metrics["runtime_failures"]==0}
 decision="GO_INTERFACE_AFFORDANCE_C1" if all(checks.values()) else "NO_GO_INTERFACE_AFFORDANCE_C1";summary={"protocol_id":protocol["protocol_id"],"decision":decision,"checks":checks,"effects":effects,"absolute":{"v6_h1_nll":macro(h1b,"action_nll"),"candidate_h1_nll":macro(h1c,"action_nll"),"v6_h1_accuracy":macro(h1b,"action_correct"),"candidate_h1_accuracy":macro(h1c,"action_correct"),"v6_h2_h5_nll":macro(mb,"action_nll"),"candidate_h2_h5_nll":macro(mc,"action_nll"),"v6_future_joint_ce":macro(jb,"joint_ce"),"candidate_future_joint_ce":macro(jc,"joint_ce")},"predictions_sha256":file_sha256(a.predictions),"run_metrics_sha256":file_sha256(a.run_metrics)}
 a.output.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");a.markdown.write_text("# Interface affordance C1\n\nDecision: `"+decision+"`\n\n"+"\n".join(f"- {k}: {'PASS' if v else 'FAIL'}" for k,v in checks.items())+"\n")
if __name__=="__main__":main()
