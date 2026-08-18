"""Train the v12 equal-capacity zero-graph and true-event-graph oracle arms."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from wmagentattack.action_event_graph_dynamics import (
    ActionEventGraphDynamics,trainable_parameter_count,
)
from wmagentattack.joint_outcome_auxiliary import JOINT_OUTCOME_CLASSES
from wmagentattack.multisource_suitability import file_sha256


def _load(name,filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/"scripts"/filename)
    module=importlib.util.module_from_spec(spec);assert spec.loader is not None
    spec.loader.exec_module(module);return module


v5=_load("v5","201_train_structured_joint_outcome_v5.py")
ARMS=("zero_graph_capacity_control_v12","true_event_graph_oracle_v12")


def _write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,indent=2)+"\n")


def _append(path,rows):
    with path.open("a") as stream:
        for row in rows:stream.write(json.dumps(row,sort_keys=True)+"\n")


def _seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)


def _surfaces(events,arrays,maximum_horizon=5):
    by=defaultdict(list)
    for index,event in enumerate(events):by[event["trajectory_id"]].append(index)
    output={}
    for horizon in range(1,maximum_horizon+1):
        starts=[];paths=[];legal=[];targets=[];future=[];sequences=[]
        for indices in by.values():
            indices=sorted(indices,key=lambda index:events[index]["step_id"])
            for position in range(len(indices)-horizon):
                sequence=indices[position:position+horizon+1]
                starts.append(sequence[0]);sequences.append(sequence)
                paths.append([arrays["selected"][index] for index in sequence[:-1]])
                legal.append([arrays["legal"][index] for index in sequence[:-1]])
                targets.append(arrays["selected"][sequence[-1]]);future.append(sequence[-2])
        output[horizon]={
            "starts":np.asarray(starts),"paths":np.asarray(paths),"legal":np.asarray(legal),
            "targets":np.asarray(targets),"future":np.asarray(future),"sequences":np.asarray(sequences),
        }
    return output


def _graph_array(events,graph_dataset):
    catalog=graph_dataset["feature_catalog"];feature_index={value:index for index,value in enumerate(catalog)}
    rows={row["event_id"]:row for row in graph_dataset["rows"]}
    output=np.zeros((len(events),len(catalog)),dtype=np.float32)
    for row_index,event in enumerate(events):
        graph=rows.get(event["event_id"])
        if graph is None:raise ValueError(f"missing event graph {event['event_id']}")
        for feature in graph["features"]:output[row_index,feature_index[feature]]=1.0
    return output


def _train_model(arm,teacher,events,arrays,surfaces,graphs,protocol,training_seed,device):
    cfg=protocol["oracle_sufficiency_stage"]["training"]
    states=torch.tensor(arrays["states"],dtype=torch.float32,device=device)
    candidates=torch.tensor(arrays["candidate_inputs"],dtype=torch.float32,device=device)
    selected=torch.tensor(arrays["selected"],dtype=torch.long,device=device)
    graph_tensor=torch.tensor(graphs,dtype=torch.float32,device=device)
    teacher.eval()
    for parameter in teacher.parameters():parameter.requires_grad_(False)
    with torch.no_grad():
        context=teacher.encode_context(states,candidates[selected])
        teacher_logits=teacher.score_candidates(context,candidates)
    _seed(training_seed*22403)
    model=ActionEventGraphDynamics(
        graph_size=graphs.shape[1],candidate_size=candidates.shape[1],
        hidden_size=cfg["hidden_size"],dropout=cfg["dropout"],
    ).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    use_graph=arm=="true_event_graph_oracle_v12";history=[]
    zeros=torch.zeros_like(graph_tensor)
    model_graph=graph_tensor if use_graph else zeros
    for epoch in range(cfg["epochs"]):
        _seed(training_seed*22403+epoch);model.train();surface=surfaces[1]
        keep=np.asarray([events[index]["split"]=="training" for index in surface["starts"]])
        starts_np=surface["starts"][keep];starts=torch.tensor(starts_np,device=device)
        targets=torch.tensor(surface["targets"][keep],device=device)
        legal=torch.tensor(surface["legal"][keep,-1],dtype=torch.bool,device=device)
        hidden=model.condition(context[starts],model_graph[starts])
        logits=(teacher_logits[starts]+model.one_step_delta_logits(hidden,candidates)).masked_fill(
            ~legal,torch.finfo(torch.float32).min
        )
        base=teacher_logits[starts].masked_fill(~legal,torch.finfo(torch.float32).min)
        weights=torch.tensor(v5._task_weights([events[index] for index in starts_np]),device=device)
        action_ce=(F.cross_entropy(logits,targets,reduction="none")*weights).sum()/weights.sum()
        base_probability=F.softmax(base,dim=1)
        kl=(base_probability*(F.log_softmax(base,dim=1)-F.log_softmax(logits,dim=1))).sum(1)
        kl=(kl*weights).sum()/weights.sum()
        total=cfg["h1_ce_weight"]*action_ce+cfg["h1_kl_weight"]*kl
        parts={"h1_ce":action_ce,"h1_kl":kl}
        for horizon in range(2,6):
            surface=surfaces[horizon]
            keep=np.asarray([events[index]["split"]=="training" for index in surface["starts"]])
            starts_np=surface["starts"][keep];starts=torch.tensor(starts_np,device=device)
            paths=torch.tensor(surface["paths"][keep],device=device)
            sequences=torch.tensor(surface["sequences"][keep],device=device)
            hidden=model.condition(context[starts],model_graph[starts])
            for step in range(1,horizon):
                hidden=model.advance(hidden,candidates[paths[:,step]],model_graph[sequences[:,step]])
            legal=torch.tensor(surface["legal"][keep,-1],dtype=torch.bool,device=device)
            targets=torch.tensor(surface["targets"][keep],device=device)
            rollout=model.rollout_logits(hidden,candidates).masked_fill(~legal,torch.finfo(torch.float32).min)
            weights=torch.tensor(v5._task_weights([events[index] for index in starts_np]),device=device)
            horizon_ce=(F.cross_entropy(rollout,targets,reduction="none")*weights).sum()/weights.sum()
            future=torch.tensor(surface["future"][keep],device=device)
            latent=1-F.cosine_similarity(model.projected_context(hidden),context[future],dim=1)
            latent=(latent*weights).sum()/weights.sum()
            trainable=np.asarray([events[index]["joint_outcome_trainable"] for index in starts_np])
            positions=np.flatnonzero(trainable);joint_loss=torch.zeros((),device=device)
            if len(positions):
                pos=torch.tensor(positions,device=device)
                target=torch.tensor(np.stack([
                    [events[starts_np[index]]["joint_outcome_target"][name] for name in JOINT_OUTCOME_CLASSES]
                    for index in positions
                ]),dtype=torch.float32,device=device)
                joint_loss=-(target*F.log_softmax(model.joint_logits(hidden[pos]),dim=1)).sum(1).mean()
            total=total+cfg["horizon_weights"][str(horizon)]*horizon_ce+cfg["latent_weight"]*latent+cfg["future_joint_weight"]*joint_loss
            parts[f"h{horizon}_ce"]=horizon_ce
        optimizer.zero_grad(set_to_none=True);total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step()
        if epoch in (0,cfg["epochs"]-1):
            history.append({"epoch":epoch,"total":float(total.detach()),"graph_gate":float(torch.tanh(model.graph_gate).detach()),**{name:float(value.detach()) for name,value in parts.items()}})
    return model,context,teacher_logits,history


def _evaluate(model,teacher,context,teacher_logits,events,arrays,surfaces,graphs,arm,fold,training_seed,device):
    candidates=torch.tensor(arrays["candidate_inputs"],dtype=torch.float32,device=device)
    graph_tensor=torch.tensor(graphs,dtype=torch.float32,device=device)
    if arm=="zero_graph_capacity_control_v12":graph_tensor=torch.zeros_like(graph_tensor)
    rows=[];model.eval()
    with torch.no_grad():
        for horizon in range(1,6):
            surface=surfaces[horizon]
            keep=np.asarray([events[index]["split"]=="confirmation" for index in surface["starts"]])
            starts_np=surface["starts"][keep];starts=torch.tensor(starts_np,device=device)
            sequences=torch.tensor(surface["sequences"][keep],device=device)
            legal_np=surface["legal"][keep]
            hidden=model.condition(context[starts],graph_tensor[starts])
            if horizon==1:
                logits=teacher_logits[starts]+model.one_step_delta_logits(hidden,candidates)
                probabilities=F.softmax(logits.masked_fill(~torch.tensor(legal_np[:,-1],dtype=torch.bool,device=device),torch.finfo(torch.float32).min),dim=1)
            else:
                probabilities=F.softmax(teacher_logits[starts].masked_fill(~torch.tensor(legal_np[:,0],dtype=torch.bool,device=device),torch.finfo(torch.float32).min),dim=1)
                for step in range(1,horizon):
                    hidden=model.advance(hidden,probabilities@candidates,graph_tensor[sequences[:,step]])
                    logits=model.rollout_logits(hidden,candidates)
                    probabilities=F.softmax(logits.masked_fill(~torch.tensor(legal_np[:,step],dtype=torch.bool,device=device),torch.finfo(torch.float32).min),dim=1)
            probability=probabilities.cpu().numpy();targets=surface["targets"][keep]
            joint_probability=torch.softmax(model.joint_logits(hidden),dim=1).cpu().numpy()
            for position,event_index in enumerate(starts_np):
                event=events[event_index];target=int(targets[position]);prediction=int(probability[position].argmax())
                joint_target=np.asarray([event["joint_outcome_target"][name] for name in JOINT_OUTCOME_CLASSES]) if event["joint_outcome_trainable"] else None
                rows.append({
                    "arm":arm,"fold":fold,"training_seed":training_seed,"horizon":horizon,
                    "event_id":event["event_id"],"task_name":event["task_name"],
                    "trajectory_id":event["trajectory_id"],"joint_group_id":event["joint_outcome_group_id"],
                    "action_nll":float(-math.log(max(probability[position,target],1e-12))),
                    "action_correct":float(prediction==target),"legal_prediction":float(legal_np[position,-1,prediction]),
                    "joint_trainable":float(joint_target is not None),
                    "joint_ce":float(-(joint_target*np.log(np.clip(joint_probability[position],1e-12,1))).sum()) if joint_target is not None else None,
                })
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--events",type=Path,required=True);parser.add_argument("--graph-dataset",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True);parser.add_argument("--device",choices=("cpu","cuda","auto"),default="auto")
    parser.add_argument("--smoke",action="store_true");args=parser.parse_args()
    protocol=json.loads(args.protocol.read_text())
    if protocol["status"]!="data_gate_passed_oracle_stage_frozen_before_training":raise ValueError("v12 oracle protocol not frozen")
    if file_sha256(args.events)!=protocol["sources"]["events_sha256"]:raise ValueError("event hash mismatch")
    if file_sha256(args.graph_dataset)!=protocol["data_gate_result"]["dataset_sha256"]:raise ValueError("graph hash mismatch")
    source=json.loads(args.events.read_text());graphs=json.loads(args.graph_dataset.read_text())
    device="cuda" if args.device=="auto" and torch.cuda.is_available() else ("cpu" if args.device=="auto" else args.device)
    if device=="cpu":torch.set_num_threads(8)
    folds=[0] if args.smoke else list(range(protocol["oracle_sufficiency_stage"]["folds"]))
    seeds=[protocol["oracle_sufficiency_stage"]["seeds"][0]] if args.smoke else protocol["oracle_sufficiency_stage"]["seeds"]
    run_protocol=copy.deepcopy(protocol)
    if args.smoke:
        run_protocol["teacher_training_protocol"]["training"]["fixed_epochs"]=1
        run_protocol["oracle_sufficiency_stage"]["training"]["epochs"]=1
    args.output_dir.mkdir(parents=True,exist_ok=True);predictions=args.output_dir/"predictions.jsonl";predictions.write_text("")
    runs=[];parameter_counts={}
    for fold in folds:
        events=v5._fold(source,fold);arrays=v5._arrays(events,source["candidate_catalog"],128)
        surfaces=_surfaces(events,arrays);graph_array=_graph_array(events,graphs)
        for training_seed in seeds:
            _seed(training_seed);teacher_values=v5._train("structured_joint_aux",events,arrays,run_protocol["teacher_training_protocol"],training_seed,device,return_model=True)
            teacher=teacher_values[4]
            for arm in ARMS:
                model,context,teacher_logits,history=_train_model(arm,teacher,events,arrays,surfaces,graph_array,run_protocol,training_seed,device)
                parameter_counts[arm]=trainable_parameter_count(model)
                rows=_evaluate(model,teacher,context,teacher_logits,events,arrays,surfaces,graph_array,arm,fold,training_seed,device)
                _append(predictions,rows);runs.append({"fold":fold,"seed":training_seed,"arm":arm,"rows":len(rows),"history":history})
    expected=len(folds)*len(seeds)*len(ARMS)
    if len(runs)!=expected:raise ValueError("incomplete fit budget")
    if len(set(parameter_counts.values()))!=1:raise ValueError("parameter mismatch")
    _write(args.output_dir/"run_metrics.json",{
        "smoke":args.smoke,"device":device,"training_units":len(runs),"runtime_failures":0,
        "parameter_counts":parameter_counts,"parameter_match":True,"runs":runs,
        "predictions_sha256":file_sha256(predictions),
    })


if __name__=="__main__":main()
