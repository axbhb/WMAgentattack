"""Train a full-graph oracle and equal-capacity late-fusion students."""

from __future__ import annotations

import argparse, copy, importlib.util, json, math, random, sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from wmagentattack.late_fusion_distilled_dynamics import LateFusionDistilledDynamics,trainable_parameter_count
from wmagentattack.multisource_suitability import file_sha256


def _load(name,filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/"scripts"/filename);module=importlib.util.module_from_spec(spec);assert spec.loader is not None;spec.loader.exec_module(module);return module


v5=_load("v5","201_train_structured_joint_outcome_v5.py");v12=_load("v12","224_train_action_event_graph_oracle_v12.py")
ARMS=("late_fusion_capacity_control_v15","late_fusion_distilled_v15")


def _seed(value):random.seed(value);np.random.seed(value);torch.manual_seed(value)
def _write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True,indent=2)+"\n")
def _append(path,rows):
    with path.open("a") as stream:
        for row in rows:stream.write(json.dumps(row,sort_keys=True)+"\n")
def _weighted(value,weight):return (value*weight).sum()/weight.sum()


def _partition_masks(graph_dataset,partition):
    catalog=graph_dataset["feature_catalog"]
    if catalog!=partition["full_feature_catalog"]:raise ValueError("catalog mismatch")
    exact=set(partition["exact_feature_catalog"]);evidence=set(partition["evidence_feature_catalog"])
    return np.asarray([x in exact for x in catalog],np.float32),np.asarray([x in evidence for x in catalog],np.float32)


def _oracle_rollout(model,context,teacher_logits,candidates,graphs,surface,keep,device):
    starts=torch.tensor(surface["starts"][keep],device=device);paths=torch.tensor(surface["paths"][keep],device=device);sequences=torch.tensor(surface["sequences"][keep],device=device)
    hidden=model.condition(context[starts],graphs[starts]);horizon=surface["paths"].shape[1]
    if horizon==1:return teacher_logits[starts]+model.one_step_delta_logits(hidden,candidates),hidden
    for step in range(1,horizon):hidden=model.advance(hidden,candidates[paths[:,step]],graphs[sequences[:,step]])
    return model.rollout_logits(hidden,candidates),hidden


def _train_student(arm,oracle,structured_teacher,events,arrays,surfaces,full_graph,exact_graph,evidence_graph,protocol,seed,device):
    cfg=protocol["training"];states=torch.tensor(arrays["states"],dtype=torch.float32,device=device);candidates=torch.tensor(arrays["candidate_inputs"],dtype=torch.float32,device=device);selected=torch.tensor(arrays["selected"],dtype=torch.long,device=device)
    full=torch.tensor(full_graph,dtype=torch.float32,device=device);exact=torch.tensor(exact_graph,dtype=torch.float32,device=device);evidence=torch.tensor(evidence_graph,dtype=torch.float32,device=device)
    structured_teacher.eval();oracle.eval()
    for module in (structured_teacher,oracle):
        for parameter in module.parameters():parameter.requires_grad_(False)
    with torch.no_grad():context=structured_teacher.encode_context(states,candidates[selected]);teacher_logits=structured_teacher.score_candidates(context,candidates)
    _seed(seed*23203);model=LateFusionDistilledDynamics(graph_size=full.shape[1],candidate_size=candidates.shape[1],hidden_size=cfg["hidden_size"],latent_size=cfg["latent_size"],dropout=cfg["dropout"]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"]);distilled=arm=="late_fusion_distilled_v15";history=[]
    for epoch in range(cfg["epochs"]):
        _seed(seed*23203+epoch);model.train();surface=surfaces[1];keep=np.asarray([events[i]["split"]=="training" for i in surface["starts"]]);starts_np=surface["starts"][keep];starts=torch.tensor(starts_np,device=device);targets=torch.tensor(surface["targets"][keep],device=device);legal=torch.tensor(surface["legal"][keep,-1],dtype=torch.bool,device=device);weights=torch.tensor(v5._task_weights([events[i] for i in starts_np]),device=device)
        hidden=model.condition(context[starts],exact[starts],evidence[starts]);logits=(teacher_logits[starts]+model.one_step_delta_logits(hidden,candidates)).masked_fill(~legal,torch.finfo(torch.float32).min);base=teacher_logits[starts].masked_fill(~legal,torch.finfo(torch.float32).min)
        action=_weighted(F.cross_entropy(logits,targets,reduction="none"),weights);base_p=F.softmax(base,1);kl=_weighted((base_p*(F.log_softmax(base,1)-F.log_softmax(logits,1))).sum(1),weights);total=cfg["h1_ce_weight"]*action+cfg["h1_kl_weight"]*kl;parts={"h1_ce":action,"h1_kl":kl}
        for horizon in range(2,6):
            surface=surfaces[horizon];keep=np.asarray([events[i]["split"]=="training" for i in surface["starts"]]);starts_np=surface["starts"][keep];starts=torch.tensor(starts_np,device=device);paths=torch.tensor(surface["paths"][keep],device=device);hidden=model.condition(context[starts],exact[starts],evidence[starts])
            for step in range(1,horizon):hidden,_=model.advance_latent(hidden,candidates[paths[:,step]])
            legal=torch.tensor(surface["legal"][keep,-1],dtype=torch.bool,device=device);targets=torch.tensor(surface["targets"][keep],device=device);student_logits=model.rollout_logits(hidden,candidates).masked_fill(~legal,torch.finfo(torch.float32).min);weights=torch.tensor(v5._task_weights([events[i] for i in starts_np]),device=device)
            action_loss=_weighted(F.cross_entropy(student_logits,targets,reduction="none"),weights);future=torch.tensor(surface["future"][keep],device=device);latent=_weighted(1-F.cosine_similarity(model.projected_context(hidden),context[future],dim=1),weights)
            with torch.no_grad():oracle_logits,_=_oracle_rollout(oracle,context,teacher_logits,candidates,full,surface,keep,device);oracle_logits=oracle_logits.masked_fill(~legal,torch.finfo(torch.float32).min);oracle_p=F.softmax(oracle_logits/cfg["oracle_temperature"],1)
            oracle_kl=_weighted((oracle_p*(F.log_softmax(oracle_logits/cfg["oracle_temperature"],1)-F.log_softmax(student_logits/cfg["oracle_temperature"],1))).sum(1),weights)
            total=total+cfg["horizon_weights"][str(horizon)]*action_loss+cfg["latent_context_weight"]*latent+(cfg["oracle_action_kl_weight"]*oracle_kl if distilled else 0)
            parts[f"h{horizon}_ce"]=action_loss;parts[f"h{horizon}_oracle_kl"]=oracle_kl
        optimizer.zero_grad(set_to_none=True);total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step()
        if epoch in (0,cfg["epochs"]-1):history.append({"epoch":epoch,"total":float(total.detach()),"gates":{name:float(torch.tanh(getattr(model,name)).detach()) for name in ("exact_gate","evidence_gate","interaction_gate","latent_gate")},**{name:float(value.detach()) for name,value in parts.items()}})
    return model,context,teacher_logits,history


def _evaluate_student(model,oracle,events,arrays,surfaces,full_graph,exact_graph,evidence_graph,context,teacher_logits,arm,fold,seed,device):
    candidates=torch.tensor(arrays["candidate_inputs"],dtype=torch.float32,device=device);full=torch.tensor(full_graph,dtype=torch.float32,device=device);exact=torch.tensor(exact_graph,dtype=torch.float32,device=device);evidence=torch.tensor(evidence_graph,dtype=torch.float32,device=device);rows=[];model.eval();oracle.eval()
    with torch.no_grad():
        for horizon in range(1,6):
            surface=surfaces[horizon];keep=np.asarray([events[i]["split"]=="confirmation" for i in surface["starts"]]);starts_np=surface["starts"][keep];starts=torch.tensor(starts_np,device=device);legal_np=surface["legal"][keep];hidden=model.condition(context[starts],exact[starts],evidence[starts])
            if horizon==1:logits=teacher_logits[starts]+model.one_step_delta_logits(hidden,candidates);legal=torch.tensor(legal_np[:,-1],dtype=torch.bool,device=device);probability=F.softmax(logits.masked_fill(~legal,torch.finfo(torch.float32).min),1)
            else:
                legal=torch.tensor(legal_np[:,0],dtype=torch.bool,device=device);probability=F.softmax(teacher_logits[starts].masked_fill(~legal,torch.finfo(torch.float32).min),1)
                for step in range(1,horizon):hidden,_=model.advance_latent(hidden,probability@candidates);legal=torch.tensor(legal_np[:,step],dtype=torch.bool,device=device);probability=F.softmax(model.rollout_logits(hidden,candidates).masked_fill(~legal,torch.finfo(torch.float32).min),1)
            oracle_logits,_=_oracle_rollout(oracle,context,teacher_logits,candidates,full,surface,keep,device);oracle_legal=torch.tensor(legal_np[:,-1],dtype=torch.bool,device=device);oracle_probability=F.softmax(oracle_logits.masked_fill(~oracle_legal,torch.finfo(torch.float32).min),1);oracle_ce=-(oracle_probability*torch.log(probability.clamp_min(1e-12))).sum(1).cpu().numpy();p=probability.cpu().numpy();targets=surface["targets"][keep]
            for position,event_index in enumerate(starts_np):
                event=events[event_index];target=int(targets[position]);prediction=int(p[position].argmax());rows.append({"arm":arm,"fold":fold,"training_seed":seed,"horizon":horizon,"event_id":event["event_id"],"task_name":event["task_name"],"trajectory_id":event["trajectory_id"],"joint_group_id":event["joint_outcome_group_id"],"action_nll":float(-math.log(max(p[position,target],1e-12))),"action_correct":float(prediction==target),"legal_prediction":float(legal_np[position,-1,prediction]),"oracle_action_ce":float(oracle_ce[position])})
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--protocol",type=Path,required=True);parser.add_argument("--events",type=Path,required=True);parser.add_argument("--graph-dataset",type=Path,required=True);parser.add_argument("--partition",type=Path,required=True);parser.add_argument("--output-dir",type=Path,required=True);parser.add_argument("--device",choices=("cpu","cuda","auto"),default="auto");parser.add_argument("--smoke",action="store_true");args=parser.parse_args();protocol=json.loads(args.protocol.read_text())
    if protocol["status"]!="preregistered_before_training":raise ValueError("v15 protocol not frozen")
    for path,key in ((args.events,"events_sha256"),(args.graph_dataset,"event_graph_sha256"),(args.partition,"partition_sha256")):
        if file_sha256(path)!=protocol["sources"][key]:raise ValueError(f"hash mismatch {key}")
    source=json.loads(args.events.read_text());graphs=json.loads(args.graph_dataset.read_text());partition=json.loads(args.partition.read_text());exact_mask,evidence_mask=_partition_masks(graphs,partition);device="cuda" if args.device=="auto" and torch.cuda.is_available() else ("cpu" if args.device=="auto" else args.device)
    if device=="cpu":torch.set_num_threads(8)
    folds=[0] if args.smoke else list(range(protocol["budget"]["folds"]));seeds=[protocol["budget"]["seeds"][0]] if args.smoke else protocol["budget"]["seeds"];run_protocol=copy.deepcopy(protocol);oracle_training=copy.deepcopy(protocol["training"]);oracle_training["latent_weight"]=oracle_training["latent_context_weight"];run_protocol["oracle_sufficiency_stage"]={"training":oracle_training}
    if args.smoke:run_protocol["teacher_training_protocol"]["training"]["fixed_epochs"]=1;run_protocol["training"]["epochs"]=1;run_protocol["oracle_sufficiency_stage"]["training"]["epochs"]=1
    args.output_dir.mkdir(parents=True,exist_ok=True);pred=args.output_dir/"predictions.jsonl";pred.write_text("");runs=[];parameter_counts={};teacher_fits=oracle_fits=0
    for fold in folds:
        events=v5._fold(source,fold);arrays=v5._arrays(events,source["candidate_catalog"],128);surfaces=v12._surfaces(events,arrays);full_graph=v12._graph_array(events,graphs);exact_graph=full_graph*exact_mask[None,:];evidence_graph=full_graph*evidence_mask[None,:]
        for seed in seeds:
            _seed(seed);values=v5._train("structured_joint_aux",events,arrays,run_protocol["teacher_training_protocol"],seed,device,return_model=True);structured=values[4];teacher_fits+=1
            oracle,oracle_context,oracle_teacher_logits,oracle_history=v12._train_model("true_event_graph_oracle_v12",structured,events,arrays,surfaces,full_graph,run_protocol,seed,device);oracle_fits+=1
            oracle_rows=v12._evaluate(oracle,structured,oracle_context,oracle_teacher_logits,events,arrays,surfaces,full_graph,"true_event_graph_oracle_v12",fold,seed,device)
            for row in oracle_rows:row["arm"]="full_graph_oracle_v15"
            _append(pred,oracle_rows);runs.append({"fold":fold,"seed":seed,"arm":"full_graph_oracle_v15","rows":len(oracle_rows),"history":oracle_history})
            for arm in ARMS:
                model,context,teacher_logits,history=_train_student(arm,oracle,structured,events,arrays,surfaces,full_graph,exact_graph,evidence_graph,run_protocol,seed,device);parameter_counts[arm]=trainable_parameter_count(model);rows=_evaluate_student(model,oracle,events,arrays,surfaces,full_graph,exact_graph,evidence_graph,context,teacher_logits,arm,fold,seed,device);_append(pred,rows);runs.append({"fold":fold,"seed":seed,"arm":arm,"rows":len(rows),"history":history})
    if len(runs)!=len(folds)*len(seeds)*3:raise ValueError("incomplete budget")
    if len(set(parameter_counts.values()))!=1:raise ValueError("parameter mismatch")
    _write(args.output_dir/"run_metrics.json",{"smoke":args.smoke,"device":device,"training_units":len(runs),"teacher_fits":teacher_fits,"oracle_fits":oracle_fits,"runtime_failures":0,"parameter_counts":parameter_counts,"parameter_match":True,"runs":runs,"predictions_sha256":file_sha256(pred)})


if __name__=="__main__":main()
