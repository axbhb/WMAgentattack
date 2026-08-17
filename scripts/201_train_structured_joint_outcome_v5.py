"""Paired Structured Markov baseline versus four-cell auxiliary loss."""

from __future__ import annotations

import argparse, json, math, random, sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.adjacent_transition import OBSERVED_OUTCOME_TARGETS, ObservedAdjacentTransitionModel
from wmagentattack.hybrid_semantic_world_model import tool_candidate_vector
from wmagentattack.joint_outcome_auxiliary import JOINT_OUTCOME_CLASSES, StructuredJointOutcomeModel, normalized_joint_event_weights
from wmagentattack.multisource_suitability import file_sha256, representation_vector

ARMS = ("structured_baseline", "structured_joint_aux")


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)+"\n", encoding="utf-8")


def _append(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False, sort_keys=True)+"\n")


def _task_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    counts=Counter(str(r["task_key"]) for r in rows)
    x=np.asarray([1/len(counts)/counts[str(r["task_key"])] for r in rows],dtype=np.float32)
    return x*len(x)/x.sum()


def _fold(dataset: Mapping[str, Any], fold: int) -> list[dict[str, Any]]:
    surface=dataset["folds"][fold]; train=set(surface["train_tasks"]); test=set(surface["test_tasks"])
    out=[]
    for source in dataset["events"]:
        task=str(source["task_name"])
        if task not in train|test: continue
        row=dict(source); row["split"]="training" if task in train else "confirmation"; out.append(row)
    if {r["task_name"] for r in out if r["split"]=="training"} != train: raise ValueError("training task mismatch")
    if {r["task_name"] for r in out if r["split"]=="confirmation"} != test: raise ValueError("confirmation task mismatch")
    return out


def _arrays(events, catalog, dimension):
    candidates=sorted(catalog); ci={c:i for i,c in enumerate(candidates)}
    states=np.stack([representation_vector(e,variant="structured_markov_v3",hash_dimension=dimension) for e in events])
    candidate_inputs=np.stack([tool_candidate_vector(catalog[c],hash_dimension=dimension) for c in candidates])
    selected=np.asarray([ci[e["current_action_candidate_id"]] for e in events],dtype=np.int64)
    legal=np.zeros((len(events),len(candidates)),bool); target=np.full(len(events),-1,np.int64)
    outcomes=np.zeros((len(events),len(OBSERVED_OUTCOME_TARGETS)),np.float32)
    joint=np.full((len(events),len(JOINT_OUTCOME_CLASSES)),np.nan,np.float32)
    for i,e in enumerate(events):
        for c in e["next_legal_candidate_ids"]: legal[i,ci[c]]=True
        if e["next_target_candidate_id"] is not None: target[i]=ci[e["next_target_candidate_id"]]
        outcomes[i]=[float(e["observed_outcome"][n]) for n in OBSERVED_OUTCOME_TARGETS]
        if e["joint_outcome_trainable"]: joint[i]=[float(e["joint_outcome_target"][n]) for n in JOINT_OUTCOME_CLASSES]
    return {"candidates":candidates,"states":states,"candidate_inputs":candidate_inputs,"selected":selected,"legal":legal,"target":target,"outcomes":outcomes,"joint":joint}


def _train(arm, events, a, protocol, seed, device, return_model=False):
    train=np.asarray([i for i,e in enumerate(events) if e["split"]=="training"],np.int64)
    tail=np.asarray([i for i in train if a["target"][i]>=0],np.int64)
    joint_idx=np.asarray([i for i in train if e_trainable(events[i])],np.int64)
    states=torch.tensor(a["states"],dtype=torch.float32,device=device); cand=torch.tensor(a["candidate_inputs"],dtype=torch.float32,device=device)
    selected=torch.tensor(a["selected"],dtype=torch.long,device=device); legal=torch.tensor(a["legal"],dtype=torch.bool,device=device)
    targets=torch.tensor(a["target"],dtype=torch.long,device=device); outcomes=torch.tensor(a["outcomes"],dtype=torch.float32,device=device)
    joint=torch.tensor(np.nan_to_num(a["joint"]),dtype=torch.float32,device=device)
    cfg=protocol["training"]
    cls=ObservedAdjacentTransitionModel if arm=="structured_baseline" else StructuredJointOutcomeModel
    model=cls(state_size=states.shape[1],candidate_size=cand.shape[1],hidden_size=cfg["hidden_size"],dropout=cfg["dropout"]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    tw=torch.tensor(_task_weights([events[i] for i in tail]),dtype=torch.float32,device=device)
    aw=torch.tensor(_task_weights([events[i] for i in train]),dtype=torch.float32,device=device)
    jw=torch.tensor(normalized_joint_event_weights(events,joint_idx),dtype=torch.float32,device=device)
    pos=a["outcomes"][train].sum(0); neg=len(train)-pos
    posw=torch.tensor(np.minimum(neg/np.maximum(pos,1),cfg["outcome_positive_weight_cap"]),dtype=torch.float32,device=device)
    history=[]
    for epoch in range(cfg["fixed_epochs"]):
        _seed(seed*1009+epoch); model.train()
        output=model(states,cand[selected],cand)
        logits,out_logits=output[:2]
        masked=logits[tail].masked_fill(~legal[tail],torch.finfo(logits.dtype).min)
        per=F.cross_entropy(masked,targets[tail],reduction="none"); action=(per*tw).sum()/tw.sum()
        per_o=F.binary_cross_entropy_with_logits(out_logits[train],outcomes[train],reduction="none",pos_weight=posw).mean(1)
        outcome=(per_o*aw).sum()/aw.sum(); joint_loss=torch.zeros((),device=device)
        if arm=="structured_joint_aux":
            logp=F.log_softmax(output[2][joint_idx],dim=1)
            per_j=-(joint[joint_idx]*logp).sum(1); joint_loss=(per_j*jw).sum()/jw.sum()
        loss=action+cfg["outcome_loss_weight"]*outcome+cfg["joint_loss_weight"]*joint_loss
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if epoch in (0,cfg["fixed_epochs"]-1): history.append({"epoch":epoch,"total":float(loss.detach().cpu()),"action":float(action.detach().cpu()),"outcome":float(outcome.detach().cpu()),"joint":float(joint_loss.detach().cpu())})
    model.eval()
    with torch.no_grad():
        output=model(states,cand[selected],cand); logits,out_logits=output[:2]
        probs=torch.softmax(logits.masked_fill(~legal,torch.finfo(logits.dtype).min),1).cpu().numpy()
        out_probs=torch.sigmoid(out_logits).cpu().numpy()
        joint_probs=torch.softmax(output[2],1).cpu().numpy() if arm=="structured_joint_aux" else None
    prior=(a["joint"][joint_idx]*normalized_joint_event_weights(events,joint_idx)[:,None]).sum(0)/normalized_joint_event_weights(events,joint_idx).sum()
    result=(probs,out_probs,joint_probs,{"history":history,"joint_prior":prior.tolist(),"training_rows":len(train),"joint_training_rows":len(joint_idx)})
    return (*result,model) if return_model else result


def e_trainable(event): return bool(event["joint_outcome_trainable"])


def _bce(p,y):
    p=np.clip(p,1e-12,1-1e-12); return -(y*np.log(p)+(1-y)*np.log(1-p))


def _predictions(events,a,probs,out_probs,joint_probs,diag,fold,arm,seed):
    rows=[]; prior=np.asarray(diag["joint_prior"],np.float32)
    for i,e in enumerate(events):
        if e["split"]!="confirmation": continue
        target=int(a["target"][i]); out=a["outcomes"][i]; ob=_bce(out_probs[i],out)
        row={"fold":fold,"arm":arm,"training_seed":seed,"event_id":e["event_id"],"task_name":e["task_name"],"trajectory_id":e["trajectory_id"],"joint_group_id":e["joint_outcome_group_id"],"step_id":e["step_id"],"has_next_action":float(target>=0),"action_nll":float(-math.log(max(probs[i,target],1e-12))) if target>=0 else None,"action_correct":float(probs[i].argmax()==target) if target>=0 else None,"legal_prediction":float(a["legal"][i,probs[i].argmax()]),"outcome_bce":float(ob.mean()),"joint_trainable":float(e_trainable(e))}
        if e_trainable(e):
            y=a["joint"][i]; p=joint_probs[i] if joint_probs is not None else prior
            row.update({"joint_cross_entropy":float(-(y*np.log(np.clip(p,1e-12,1))).sum()),"joint_brier":float(((p-y)**2).mean()),"joint_prior_cross_entropy":float(-(y*np.log(np.clip(prior,1e-12,1))).sum()),"joint_p11":float(p[3]),"joint_target_p11":float(y[3])})
        else: row.update({"joint_cross_entropy":None,"joint_brier":None,"joint_prior_cross_entropy":None,"joint_p11":None,"joint_target_p11":None})
        rows.append(row)
    return rows


def main():
    p=argparse.ArgumentParser(); p.add_argument("--protocol",type=Path,required=True); p.add_argument("--dataset",type=Path,required=True); p.add_argument("--audit",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--device",choices=("cpu","cuda","auto"),default="auto"); args=p.parse_args()
    protocol=json.loads(args.protocol.read_text()); frozen=protocol["frozen_dataset"]
    if protocol["status"]!="preflight_passed_and_frozen_before_training": raise ValueError("protocol not frozen")
    if file_sha256(args.dataset)!=frozen["sha256"] or file_sha256(args.audit)!=frozen["audit_sha256"]: raise ValueError("frozen data hash mismatch")
    if not json.loads(args.audit.read_text())["passed"]: raise ValueError("audit failed")
    data=json.loads(args.dataset.read_text()); seeds=protocol["training"]["training_seeds"]
    device="cuda" if args.device=="auto" and torch.cuda.is_available() else ("cpu" if args.device=="auto" else args.device)
    if device=="cpu": torch.set_num_threads(8)
    args.output_dir.mkdir(parents=True,exist_ok=True); pred=args.output_dir/"predictions.jsonl"; pred.write_text("")
    runs=[]
    for fold in range(5):
        events=_fold(data,fold); a=_arrays(events,data["candidate_catalog"],protocol["training"]["hash_dimension"])
        for arm in ARMS:
            for seed in seeds:
                _seed(seed); values=_train(arm,events,a,protocol,seed,device)
                rows=_predictions(events,a,*values[:3],values[3],fold,arm,seed); _append(pred,rows)
                runs.append({"fold":fold,"arm":arm,"seed":seed,"prediction_rows":len(rows),**values[3]})
    metrics={"neural_training_runs":len(runs),"device":device,"runs":runs,"predictions_sha256":file_sha256(pred),"runtime_failures":0}
    if len(runs)!=protocol["fixed_budget"]["neural_training_runs"]: raise ValueError("budget incomplete")
    _write(args.output_dir/"run_metrics.json",metrics)


if __name__=="__main__": main()
