"""Train zero-initialized multi-step residuals on frozen Structured+joint teachers."""
from __future__ import annotations
import argparse,copy,importlib.util,json,math,random,sys
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from wmagentattack.multisource_suitability import file_sha256
from wmagentattack.joint_outcome_auxiliary import JOINT_OUTCOME_CLASSES,normalized_joint_event_weights
from wmagentattack.structured_residual_dynamics import StructuredResidualDynamics

spec=importlib.util.spec_from_file_location("v5",ROOT/"scripts/201_train_structured_joint_outcome_v5.py"); v5=importlib.util.module_from_spec(spec); spec.loader.exec_module(v5)

def write(p,x): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,sort_keys=True,indent=2)+"\n")
def append(p,rows):
 with p.open("a") as f:
  for r in rows:f.write(json.dumps(r,sort_keys=True)+"\n")
def seed(s): random.seed(s);np.random.seed(s);torch.manual_seed(s)

def horizons(events,a,max_h=5):
 by=defaultdict(list)
 for i,e in enumerate(events):by[e["trajectory_id"]].append(i)
 out={}
 for h in range(1,max_h+1):
  starts=[];paths=[];legals=[];targets=[];future=[]
  for ids in by.values():
   ids=sorted(ids,key=lambda i:events[i]["step_id"])
   for p in range(len(ids)-h):
    seq=ids[p:p+h+1];starts.append(seq[0]);paths.append([a["selected"][i] for i in seq[:-1]]);legals.append([a["legal"][i] for i in seq[:-1]]);targets.append(a["selected"][seq[-1]]);future.append(seq[-2])
  out[h]={"starts":np.asarray(starts),"paths":np.asarray(paths),"legals":np.asarray(legals),"targets":np.asarray(targets),"future":np.asarray(future)}
 return out

def train_residual(teacher,events,a,surfaces,protocol,s,device):
 cfg=protocol["residual_training"]; teacher.eval()
 for p in teacher.parameters():p.requires_grad_(False)
 states=torch.tensor(a["states"],dtype=torch.float32,device=device);cand=torch.tensor(a["candidate_inputs"],dtype=torch.float32,device=device);selected=torch.tensor(a["selected"],dtype=torch.long,device=device)
 with torch.no_grad(): context=teacher.encode_context(states,cand[selected]); teacher_logits=teacher.score_candidates(context,cand)
 model=StructuredResidualDynamics(candidate_size=cand.shape[1],hidden_size=cfg["hidden_size"],dropout=cfg["dropout"]).to(device)
 opt=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
 history=[]
 for epoch in range(cfg["epochs"]):
  seed(s*2003+epoch);model.train();total=torch.zeros((),device=device);parts={}
  sf=surfaces[1];keep=np.asarray([events[i]["split"]=="training" for i in sf["starts"]]);idx=torch.tensor(sf["starts"][keep],device=device);tgt=torch.tensor(sf["targets"][keep],device=device);legal=torch.tensor(sf["legals"][keep,-1],dtype=torch.bool,device=device)
  base=teacher_logits[idx].masked_fill(~legal,torch.finfo(torch.float32).min); logits=(teacher_logits[idx]+model.one_step_delta_logits(context[idx],cand)).masked_fill(~legal,torch.finfo(torch.float32).min)
  w=torch.tensor(v5._task_weights([events[i] for i in sf["starts"][keep]]),device=device);ce=(F.cross_entropy(logits,tgt,reduction="none")*w).sum()/w.sum();bp=F.softmax(base,1);kl=(bp*(F.log_softmax(base,1)-F.log_softmax(logits,1))).sum(1);kl=(kl*w).sum()/w.sum();total=cfg["h1_ce_weight"]*ce+cfg["h1_kl_weight"]*kl;parts.update(h1_ce=ce,h1_kl=kl)
  for h in range(2,6):
   sf=surfaces[h];keep=np.asarray([events[i]["split"]=="training" for i in sf["starts"]]);starts=torch.tensor(sf["starts"][keep],device=device);paths=torch.tensor(sf["paths"][keep],device=device);hidden=context[starts]
   for k in range(1,h):hidden=model.advance(hidden,cand[paths[:,k]])
   legal=torch.tensor(sf["legals"][keep,-1],dtype=torch.bool,device=device);tgt=torch.tensor(sf["targets"][keep],device=device);logits=model.rollout_logits(hidden,cand).masked_fill(~legal,torch.finfo(torch.float32).min);w=torch.tensor(v5._task_weights([events[i] for i in sf["starts"][keep]]),device=device);hce=(F.cross_entropy(logits,tgt,reduction="none")*w).sum()/w.sum();future=torch.tensor(sf["future"][keep],device=device);latent=(1-F.cosine_similarity(model.projected_context(hidden),context[future],dim=1));latent=(latent*w).sum()/w.sum()
   trainable=np.asarray([events[i]["joint_outcome_trainable"] for i in sf["starts"][keep]]);ji=torch.tensor(np.flatnonzero(trainable),device=device);jloss=torch.zeros((),device=device)
   if len(ji):
    y=torch.tensor(np.stack([[events[sf["starts"][keep][i]]["joint_outcome_target"][c] for c in JOINT_OUTCOME_CLASSES] for i in np.flatnonzero(trainable)]),dtype=torch.float32,device=device);jloss=-(y*F.log_softmax(model.joint_logits(hidden[ji]),1)).sum(1).mean()
   total=total+cfg["horizon_weights"][str(h)]*hce+cfg["latent_weight"]*latent+cfg["future_joint_weight"]*jloss;parts[f"h{h}_ce"]=hce
  opt.zero_grad();total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);opt.step()
  if epoch in (0,cfg["epochs"]-1):history.append({"epoch":epoch,"total":float(total.detach()),**{k:float(v.detach()) for k,v in parts.items()}})
 return model,context,teacher_logits,history

def evaluate(model,teacher,context,teacher_logits,events,a,surfaces,diag,fold,s,device):
 cand=torch.tensor(a["candidate_inputs"],dtype=torch.float32,device=device);rows=[];prior=np.asarray(diag["joint_prior"])
 model.eval()
 with torch.no_grad():
  for arm in ["structured_joint_teacher","structured_residual_v6"]:
   for h in range(1,6):
    if arm=="structured_joint_teacher" and h>1:continue
    sf=surfaces[h];keep=np.asarray([events[i]["split"]=="confirmation" for i in sf["starts"]]);starts_np=sf["starts"][keep];starts=torch.tensor(starts_np,device=device);legal_np=sf["legals"][keep]
    if h==1:
     logits=teacher_logits[starts]
     if arm=="structured_residual_v6":logits=logits+model.one_step_delta_logits(context[starts],cand)
     probs=F.softmax(logits.masked_fill(~torch.tensor(legal_np[:,-1],dtype=torch.bool,device=device),torch.finfo(torch.float32).min),1);hidden=context[starts]
    else:
     hidden=context[starts];base=teacher_logits[starts];probs=F.softmax(base.masked_fill(~torch.tensor(legal_np[:,0],dtype=torch.bool,device=device),torch.finfo(torch.float32).min),1)
     for k in range(1,h):
      hidden=model.advance(hidden,probs@cand);logits=model.rollout_logits(hidden,cand);probs=F.softmax(logits.masked_fill(~torch.tensor(legal_np[:,k],dtype=torch.bool,device=device),torch.finfo(torch.float32).min),1)
    probs=probs.cpu().numpy();targets=sf["targets"][keep]
    jprob=(torch.softmax(teacher.joint_outcome_head(hidden),1) if h==1 else torch.softmax(model.joint_logits(hidden),1)).cpu().numpy()
    for n,i in enumerate(starts_np):
     e=events[i];t=int(targets[n]);y=np.asarray([e["joint_outcome_target"][c] for c in JOINT_OUTCOME_CLASSES]) if e["joint_outcome_trainable"] else None
     rows.append({"arm":arm,"fold":fold,"training_seed":s,"horizon":h,"event_id":e["event_id"],"task_name":e["task_name"],"trajectory_id":e["trajectory_id"],"joint_group_id":e["joint_outcome_group_id"],"action_nll":float(-math.log(max(probs[n,t],1e-12))),"action_correct":float(probs[n].argmax()==t),"legal_prediction":float(legal_np[n,-1,probs[n].argmax()]),"joint_trainable":float(y is not None),"joint_ce":float(-(y*np.log(np.clip(jprob[n],1e-12,1))).sum()) if y is not None else None,"joint_prior_ce":float(-(y*np.log(np.clip(prior,1e-12,1))).sum()) if y is not None else None})
 return rows

def main():
 p=argparse.ArgumentParser();p.add_argument("--protocol",type=Path,required=True);p.add_argument("--dataset",type=Path,required=True);p.add_argument("--audit",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);args=p.parse_args();protocol=json.loads(args.protocol.read_text());data=json.loads(args.dataset.read_text());device="cpu";torch.set_num_threads(8)
 if file_sha256(args.dataset)!=protocol["frozen_dataset"]["sha256"]:raise ValueError("data hash")
 args.output_dir.mkdir(parents=True,exist_ok=True);pred=args.output_dir/"predictions.jsonl";pred.write_text("");runs=[]
 v5protocol=copy.deepcopy(protocol["teacher_training_protocol"])
 for fold in range(5):
  events=v5._fold(data,fold);a=v5._arrays(events,data["candidate_catalog"],128);sf=horizons(events,a)
  for s in [7,17,29]:
   seed(s);values=v5._train("structured_joint_aux",events,a,v5protocol,s,device,return_model=True);teacher=values[4]
   model,context,logits,history=train_residual(teacher,events,a,sf,protocol,s,device);rows=evaluate(model,teacher,context,logits,events,a,sf,values[3],fold,s,device);append(pred,rows);runs.append({"fold":fold,"seed":s,"rows":len(rows),"teacher_history":values[3]["history"],"residual_history":history})
 metrics={"training_units":len(runs),"predictions_sha256":file_sha256(pred),"runs":runs,"runtime_failures":0};write(args.output_dir/"run_metrics.json",metrics)
if __name__=="__main__":main()
