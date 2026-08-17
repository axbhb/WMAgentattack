"""Gate the Structured joint + zero-residual v6 experiment."""
from __future__ import annotations
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from wmagentattack.multisource_suitability import file_sha256
from wmagentattack.multisource_suitability_experiment import paired_bootstrap,exact_sign_test
def read(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def tmap(rows,metric):
 d=defaultdict(list)
 for r in rows:d[r["task_name"]].append(float(r[metric]))
 return {k:np.mean(v) for k,v in d.items()}
def effect(left,right,metric,higher=False):
 out={}
 for s in [7,17,29]:
  a=tmap([r for r in left if int(r["training_seed"])==s],metric);b=tmap([r for r in right if int(r["training_seed"])==s],metric);out[s]={t:(b[t]-a[t] if higher else a[t]-b[t]) for t in a}
 tasks=set(out[7]);paired={t:float(np.mean([out[s][t] for s in out])) for t in tasks};seeds={str(s):float(np.mean(list(out[s].values()))) for s in out}
 return {"mean":float(np.mean(list(seeds.values()))),"seeds":seeds,"tasks":paired,"positive":sum(v>0 for v in paired.values())/len(paired),"bootstrap":paired_bootstrap(list(paired.values()),draws=10000,seed=8172600+len(metric)),"sign":exact_sign_test(list(paired.values()))}
def main():
 p=argparse.ArgumentParser();p.add_argument("--protocol",type=Path,required=True);p.add_argument("--predictions",type=Path,required=True);p.add_argument("--run-metrics",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--markdown",type=Path,required=True);a=p.parse_args();pr=json.loads(a.protocol.read_text());cur=read(a.predictions);v5=read(pr["external_controls"]["v5_predictions"]);v4=read(pr["external_controls"]["v4_predictions"])
 if file_sha256(Path(pr["external_controls"]["v5_predictions"]))!=pr["external_controls"]["v5_sha256"] or file_sha256(Path(pr["external_controls"]["v4_predictions"]))!=pr["external_controls"]["v4_sha256"]:raise ValueError("external hash")
 teacher=[r for r in cur if r["arm"]=="structured_joint_teacher" and r["horizon"]==1];res1=[r for r in cur if r["arm"]=="structured_residual_v6" and r["horizon"]==1];base=[r for r in v5 if r["arm"]=="structured_baseline" and r["has_next_action"]];extteacher=[r for r in v5 if r["arm"]=="structured_joint_aux" and r["has_next_action"]]
 effects={"h1_nll_vs_original":effect(base,res1,"action_nll"),"h1_acc_vs_original":effect(base,res1,"action_correct",True),"h1_nll_vs_teacher":effect(teacher,res1,"action_nll"),"h1_acc_vs_teacher":effect(teacher,res1,"action_correct",True),"teacher_replication_nll":effect(extteacher,teacher,"action_nll")}
 multi_left=[];multi_right=[]
 for h in range(2,6):multi_left += [r for r in v4 if r["kind"]=="rollout" and r["arm"]=="fns_bwm_multihorizon" and r["horizon"]==h];multi_right += [r for r in cur if r["arm"]=="structured_residual_v6" and r["horizon"]==h]
 effects["h2_h5_nll_vs_typed_v4"]=effect(multi_left,multi_right,"action_nll")
 joint=[r for r in cur if r["arm"]=="structured_residual_v6" and r["horizon"]>=2 and r["joint_trainable"]];prior=[{**r,"joint_ce":r["joint_prior_ce"]} for r in joint];effects["future_joint_ce_vs_prior"]=effect(prior,joint,"joint_ce")
 g=pr["acceptance_gate"];checks={"h1_nll_gain":effects["h1_nll_vs_original"]["mean"]>=g["minimum_h1_nll_gain_over_original"],"h1_accuracy_gain":effects["h1_acc_vs_original"]["mean"]>=g["minimum_h1_accuracy_gain_over_original"],"h1_nll_noninferiority":effects["h1_nll_vs_teacher"]["mean"]>=-g["maximum_h1_nll_degradation_vs_teacher"],"h1_accuracy_noninferiority":effects["h1_acc_vs_teacher"]["mean"]>=-g["maximum_h1_accuracy_degradation_vs_teacher"],"teacher_replication":abs(effects["teacher_replication_nll"]["mean"])<1e-6,"h2_h5_gain":effects["h2_h5_nll_vs_typed_v4"]["mean"]>=g["minimum_h2_h5_nll_gain_over_typed_v4"],"h2_h5_positive_tasks":effects["h2_h5_nll_vs_typed_v4"]["positive"]>=g["minimum_h2_h5_positive_task_fraction"],"h2_h5_seed_replication":sum(v>=g["minimum_h2_h5_nll_gain_over_typed_v4"] for v in effects["h2_h5_nll_vs_typed_v4"]["seeds"].values())>=g["minimum_positive_seeds"],"future_joint_gain":effects["future_joint_ce_vs_prior"]["mean"]>=g["minimum_future_joint_ce_gain_over_prior"],"all_legal":all(r["legal_prediction"]==1 for r in cur)}
 decision="GO_RETAIN_STRUCTURED_JOINT_RESIDUAL_V6" if all(checks.values()) else "NO_GO_STRUCTURED_RESIDUAL_V6"
 summary={"protocol_id":pr["protocol_id"],"decision":decision,"gate_checks":checks,"effects":effects,"predictions_sha256":file_sha256(a.predictions),"run_metrics_sha256":file_sha256(a.run_metrics)};a.output.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");a.markdown.write_text("# Structured residual v6\n\nDecision: `"+decision+"`\n\n"+"\n".join(f"- {k}: {'PASS' if v else 'FAIL'}" for k,v in checks.items())+"\n")
if __name__=="__main__":main()
