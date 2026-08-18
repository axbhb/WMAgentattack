"""Apply the frozen v15 late-fusion distillation gate."""

from __future__ import annotations
import argparse,importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from wmagentattack.multisource_suitability import file_sha256


def _load(name,filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/"scripts"/filename);module=importlib.util.module_from_spec(spec);assert spec.loader is not None;spec.loader.exec_module(module);return module
summary12=_load("summary12","225_summarize_action_event_graph_oracle_v12.py")
def _read(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
def _effect(left,right,metric,**kwargs):return summary12._effect(left,right,metric,**kwargs)


def main():
    p=argparse.ArgumentParser();p.add_argument("--protocol",type=Path,required=True);p.add_argument("--predictions",type=Path,required=True);p.add_argument("--run-metrics",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--markdown",type=Path,required=True);p.add_argument("--composite",type=Path,required=True);args=p.parse_args();protocol=json.loads(args.protocol.read_text());rows=_read(args.predictions);metrics=json.loads(args.run_metrics.read_text());v6_path=Path(protocol["sources"]["v6_predictions"])
    if file_sha256(v6_path)!=protocol["sources"]["v6_predictions_sha256"]:raise ValueError("v6 hash mismatch")
    v6=[r for r in _read(v6_path) if r["arm"]=="structured_residual_v6"];control=[r for r in rows if r["arm"]=="late_fusion_capacity_control_v15"];candidate=[r for r in rows if r["arm"]=="late_fusion_distilled_v15"];oracle=[r for r in rows if r["arm"]=="full_graph_oracle_v15"]
    if not len(v6)==len(control)==len(candidate)==len(oracle):raise ValueError("paired row mismatch")
    keys=("fold","training_seed","horizon","event_id");v6_by={tuple(r[k] for k in keys):r for r in v6};composite=[]
    for row in candidate:
        source=v6_by[tuple(row[k] for k in keys)];out=dict(row);out["arm"]="late_fusion_action_v6_outcome_v15";out["joint_trainable"]=source["joint_trainable"];out["joint_ce"]=source["joint_ce"];composite.append(out)
    args.composite.write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in composite))
    h1=lambda values:[r for r in values if r["horizon"]==1];multi=lambda values:[r for r in values if r["horizon"]>=2]
    effects={
      "h1_nll_vs_v6":_effect(h1(v6),h1(candidate),"action_nll",seed=81501),
      "h1_accuracy_vs_v6":_effect(h1(v6),h1(candidate),"action_correct",higher=True,seed=81502),
      "h2_h5_nll_vs_v6":_effect(multi(v6),multi(candidate),"action_nll",seed=81503),
      "h2_h5_nll_vs_capacity":_effect(multi(control),multi(candidate),"action_nll",seed=81504),
      "full_oracle_h2_h5_vs_v6":_effect(multi(v6),multi(oracle),"action_nll",seed=81505),
      "oracle_ce_vs_capacity":_effect(multi(control),multi(candidate),"oracle_action_ce",seed=81506),
      "future_joint_ce_vs_v6":_effect([r for r in multi(v6) if r["joint_trainable"]],[r for r in multi(composite) if r["joint_trainable"]],"joint_ce",seed=81507),
    }
    gate=protocol["gate"];gain=effects["h2_h5_nll_vs_v6"]["mean"];full_gain=effects["full_oracle_h2_h5_vs_v6"]["mean"]
    checks={"complete_budget":metrics["training_units"]==45 and metrics["teacher_fits"]==15 and metrics["oracle_fits"]==15,"runtime_clean":metrics["runtime_failures"]==0,"parameter_match":bool(metrics["parameter_match"]),"paired_rows_complete":len(candidate)==len(v6),"h1_nll_noninferiority":effects["h1_nll_vs_v6"]["mean"]>=-gate["maximum_h1_nll_degradation_vs_v6"],"h1_accuracy_noninferiority":effects["h1_accuracy_vs_v6"]["mean"]>=-gate["maximum_h1_accuracy_degradation_vs_v6"],"h2_h5_gain_vs_v6":gain>=gate["minimum_h2_h5_nll_gain_vs_v6"],"h2_h5_gain_vs_capacity":effects["h2_h5_nll_vs_capacity"]["mean"]>=gate["minimum_h2_h5_nll_gain_vs_capacity_control"],"fraction_of_full_oracle":full_gain>0 and gain/full_gain>=gate["minimum_fraction_of_v14_full_oracle_gain"],"task_breadth":effects["h2_h5_nll_vs_v6"]["positive_task_fraction"]>=gate["minimum_task_positive_fraction"],"seed_replication":sum(v>0 for v in effects["h2_h5_nll_vs_v6"]["seeds"].values())>=gate["minimum_positive_seeds"],"oracle_kl_gain":effects["oracle_ce_vs_capacity"]["mean"]>=gate["minimum_oracle_kl_gain_vs_capacity"],"future_joint_exact":abs(effects["future_joint_ce_vs_v6"]["mean"])<=gate["maximum_absolute_future_joint_ce_difference_vs_v6"],"all_legal":all(r["legal_prediction"]==1 for r in rows)}
    decision="GO_LATE_FUSION_DISTILLED_WORLD_MODEL_V15" if all(checks.values()) else "NO_GO_LATE_FUSION_DISTILLED_WORLD_MODEL_V15";result={"protocol_id":protocol["protocol_id"],"decision":decision,"gate_checks":checks,"failed_checks":[k for k,v in checks.items() if not v],"effects":effects,"retained_fraction_of_full_oracle":gain/full_gain if full_gain else None,"counts":{"v6":len(v6),"control":len(control),"candidate":len(candidate),"oracle":len(oracle)},"predictions_sha256":file_sha256(args.predictions),"composite_sha256":file_sha256(args.composite),"run_metrics_sha256":file_sha256(args.run_metrics)};args.output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    lines=["# Late-fusion distillation v15","",f"Decision: `{decision}`","","## Frozen gate",""]+[f"- {k}: {'PASS' if v else 'FAIL'}" for k,v in checks.items()]+["","## Effects",""]+[f"- {k}: {v['mean']:.6f}" for k,v in effects.items()]+[""];args.markdown.write_text("\n".join(lines))


if __name__=="__main__":main()
