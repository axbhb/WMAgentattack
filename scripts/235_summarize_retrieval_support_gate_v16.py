"""Apply the frozen v16 retrieval support gate."""

from __future__ import annotations
import argparse,importlib.util,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from wmagentattack.multisource_suitability import file_sha256


def _load(name,filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/"scripts"/filename);module=importlib.util.module_from_spec(spec);assert spec.loader is not None;spec.loader.exec_module(module);return module
s12=_load("s12","225_summarize_action_event_graph_oracle_v12.py")
def _read(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    p=argparse.ArgumentParser();p.add_argument("--protocol",type=Path,required=True);p.add_argument("--predictions",type=Path,required=True);p.add_argument("--run-metrics",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--markdown",type=Path,required=True);args=p.parse_args();protocol=json.loads(args.protocol.read_text());retrieval=_read(args.predictions);metrics=json.loads(args.run_metrics.read_text());v6_path=Path(protocol["sources"]["v6_predictions"])
    if file_sha256(v6_path)!=protocol["sources"]["v6_predictions_sha256"]:raise ValueError("v6 hash mismatch")
    v6=[r for r in _read(v6_path) if r["arm"]=="structured_residual_v6"]
    keys=("fold","training_seed","horizon","event_id");v6_by={tuple(r[k] for k in keys):r for r in v6};retrieval_by={tuple(r[k] for k in keys):r for r in retrieval}
    if set(v6_by)!=set(retrieval_by):raise ValueError("pairing mismatch")
    supported=[r for r in retrieval if r["horizon"]>=2 and r["supported"]];supported_keys={tuple(r[k] for k in keys) for r in supported};v6_supported=[v6_by[key] for key in supported_keys]
    effect=s12._effect(v6_supported,supported,"action_nll",seed=81601)
    multi=[r for r in retrieval if r["horizon"]>=2];coverage=len(supported)/len(multi)
    support_values=defaultdict(list)
    for row in multi:support_values[bool(row["supported"])].append(float(row["action_nll"]))
    gap=float(np.mean(support_values[False])-np.mean(support_values[True])) if support_values[False] else 0.0
    gate=protocol["gate"];checks={"complete_rows":len(retrieval)==len(v6),"runtime_clean":metrics["runtime_failures"]==0,"supported_coverage":coverage>=gate["minimum_h2_h5_supported_fraction"],"supported_gain_vs_v6":effect["mean"]>=gate["minimum_supported_h2_h5_nll_gain_vs_v6"],"supported_task_breadth":effect["positive_task_fraction"]>=gate["minimum_supported_positive_task_fraction"],"supported_seed_replication":sum(v>0 for v in effect["seeds"].values())>=gate["minimum_supported_positive_seeds"],"support_orders_error":gap>=gate["minimum_uncovered_minus_supported_nll_gap"],"all_legal":all(r["legal_prediction"]==1 for r in retrieval)}
    decision="GO_RETRIEVAL_RESIDUAL_FUSION_V16" if all(checks.values()) else "NO_GO_RETRIEVAL_SUPPORT_V16";result={"protocol_id":protocol["protocol_id"],"decision":decision,"gate_checks":checks,"failed_checks":[k for k,v in checks.items() if not v],"h2_h5_supported_fraction":coverage,"supported_h2_h5_nll_gain_vs_v6":effect,"uncovered_minus_supported_nll_gap":gap,"counts":{"retrieval":len(retrieval),"v6":len(v6),"supported_h2_h5":len(supported)},"predictions_sha256":file_sha256(args.predictions),"run_metrics_sha256":file_sha256(args.run_metrics)};args.output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n");args.markdown.write_text("# Retrieval support gate v16\n\nDecision: `"+decision+"`\n\n"+"\n".join(f"- {k}: {'PASS' if v else 'FAIL'}" for k,v in checks.items())+f"\n\n- supported fraction: {coverage:.6f}\n- supported NLL gain vs v6: {effect['mean']:.6f}\n- uncovered minus supported NLL: {gap:.6f}\n")


if __name__=="__main__":main()
