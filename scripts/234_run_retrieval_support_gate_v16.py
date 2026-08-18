"""Evaluate a fixed task-disjoint nearest-transition successor model."""

from __future__ import annotations
import argparse,importlib.util,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from wmagentattack.multisource_suitability import file_sha256


def _load(name,filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/"scripts"/filename);module=importlib.util.module_from_spec(spec);assert spec.loader is not None;spec.loader.exec_module(module);return module
v5=_load("v5","201_train_structured_joint_outcome_v5.py");v12=_load("v12","224_train_action_event_graph_oracle_v12.py")
def _normalize(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)
def _embedding(arrays,graphs,cfg):
    state=_normalize(arrays["states"])*cfg["state_block_weight"]**0.5;graph=_normalize(graphs)*cfg["event_graph_block_weight"]**0.5;action=_normalize(arrays["candidate_inputs"])[arrays["selected"]]*cfg["current_action_block_weight"]**0.5;return _normalize(np.concatenate((state,graph,action),axis=1).astype(np.float32))
def _topk(query,index,k):
    similarity=query@index.T;k=min(k,index.shape[0]);indices=np.argpartition(similarity,-k,axis=1)[:,-k:];values=np.take_along_axis(similarity,indices,axis=1);order=np.argsort(values,axis=1)[:,::-1];return np.take_along_axis(values,order,axis=1),np.take_along_axis(indices,order,axis=1)
def _calibration(embedding,indices,trajectory_ids,k):
    values=[]
    for start in range(0,len(indices),512):
        batch=indices[start:start+512];similarity=embedding[batch]@embedding[indices].T
        for row,event_index in enumerate(batch):similarity[row,np.asarray([trajectory_ids[j]==trajectory_ids[event_index] for j in indices])]=-np.inf
        values.extend(np.max(similarity,axis=1).tolist())
    return float(np.quantile(np.asarray(values),0.25))
def _write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True,indent=2)+"\n")


def main():
    p=argparse.ArgumentParser();p.add_argument("--protocol",type=Path,required=True);p.add_argument("--events",type=Path,required=True);p.add_argument("--graph-dataset",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--smoke",action="store_true");args=p.parse_args();protocol=json.loads(args.protocol.read_text())
    if protocol["status"]!="preregistered_support_gate_before_execution":raise ValueError("v16 protocol not frozen")
    if file_sha256(args.events)!=protocol["sources"]["events_sha256"] or file_sha256(args.graph_dataset)!=protocol["sources"]["event_graph_sha256"]:raise ValueError("source hash mismatch")
    source=json.loads(args.events.read_text());graph_dataset=json.loads(args.graph_dataset.read_text());cfg=protocol["retrieval"];rows=[];diagnostics=[]
    folds=[0] if args.smoke else range(protocol["budget"]["folds"]);horizons=[1] if args.smoke else range(1,6);seeds=protocol["budget"]["seeds_for_pairing"][:1] if args.smoke else protocol["budget"]["seeds_for_pairing"]
    for fold in folds:
        events=v5._fold(source,fold);arrays=v5._arrays(events,source["candidate_catalog"],128);graphs=v12._graph_array(events,graph_dataset);embedding=_embedding(arrays,graphs,cfg);surfaces=v12._surfaces(events,arrays);trajectory_ids=[event["trajectory_id"] for event in events]
        for horizon in horizons:
            surface=surfaces[horizon];train_mask=np.asarray([events[i]["split"]=="training" for i in surface["starts"]]);test_mask=~train_mask;train_starts=surface["starts"][train_mask];test_starts=surface["starts"][test_mask];train_targets=surface["targets"][train_mask];threshold=_calibration(embedding,train_starts,trajectory_ids,cfg["k"]);similarity,neighbor=_topk(embedding[test_starts],embedding[train_starts],cfg["k"]);neighbor_targets=train_targets[neighbor];legal=surface["legal"][test_mask,-1];targets=surface["targets"][test_mask];probabilities=np.zeros((len(test_starts),len(arrays["candidates"])),np.float64)
            weight=np.exp((similarity-similarity[:,0:1])/cfg["temperature"])
            for i in range(len(test_starts)):
                probabilities[i,legal[i]]=cfg["legal_smoothing"]
                for j,target in enumerate(neighbor_targets[i]):
                    if legal[i,target]:probabilities[i,target]+=weight[i,j]
                total=probabilities[i].sum();probabilities[i]=probabilities[i]/total if total else legal[i]/legal[i].sum()
            support=similarity[:,0]>=threshold;diagnostics.append({"fold":fold,"horizon":horizon,"training_anchors":len(train_starts),"confirmation_queries":len(test_starts),"support_threshold":threshold,"supported_fraction":float(support.mean())})
            for i,event_index in enumerate(test_starts):
                prediction=int(probabilities[i].argmax());base={"arm":"retrieval_successor_v16","fold":fold,"horizon":horizon,"event_id":events[event_index]["event_id"],"task_name":events[event_index]["task_name"],"trajectory_id":events[event_index]["trajectory_id"],"joint_group_id":events[event_index]["joint_outcome_group_id"],"action_nll":float(-math.log(max(probabilities[i,targets[i]],1e-12))),"action_correct":float(prediction==targets[i]),"legal_prediction":float(legal[i,prediction]),"supported":bool(support[i]),"top1_similarity":float(similarity[i,0]),"support_threshold":threshold}
                for seed in seeds:rows.append({**base,"training_seed":seed})
    args.output_dir.mkdir(parents=True,exist_ok=True);pred=args.output_dir/"predictions.jsonl";pred.write_text("".join(json.dumps(row,sort_keys=True)+"\n" for row in rows));_write(args.output_dir/"run_metrics.json",{"runtime_failures":0,"rows":len(rows),"diagnostics":diagnostics,"predictions_sha256":file_sha256(pred)})


if __name__=="__main__":main()
