"""Stage C2: interface affordances with direct successor-action supervision."""
from __future__ import annotations
import argparse,importlib.util,json,math,sys
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from wmagentattack.joint_outcome_auxiliary import JOINT_OUTCOME_CLASSES
from wmagentattack.multisource_suitability import file_sha256
from wmagentattack.relational_slot_latent import SuccessorAffordanceResidual,stack_interface_affordance_states
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
v5=load("v5",ROOT/"scripts/201_train_structured_joint_outcome_v5.py");v6=load("v6",ROOT/"scripts/203_train_structured_residual_v6.py");stage_a=load("stage_a",ROOT/"scripts/205_train_relational_slot_stage_a.py")
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
def append(path,rows):
 with path.open("a") as f:
  for row in rows:f.write(json.dumps(row,sort_keys=True)+"\n")
def train(teacher,events,arrays,surfaces,slots,protocol,training_seed,device):
 cfg=protocol["stage_c1"]["residual"];extra=protocol["stage_c2"];teacher.eval()
 for parameter in teacher.parameters():parameter.requires_grad_(False)
 states=torch.tensor(arrays["states"],dtype=torch.float32,device=device);candidates=torch.tensor(arrays["candidate_inputs"],dtype=torch.float32,device=device);selected=torch.tensor(arrays["selected"],dtype=torch.long,device=device)
 sf=torch.tensor(slots["features"],dtype=torch.float32,device=device);st=torch.tensor(slots["node_types"],dtype=torch.long,device=device);sr=torch.tensor(slots["relations"],dtype=torch.long,device=device);sm=torch.tensor(slots["mask"],dtype=torch.bool,device=device)
 with torch.no_grad():teacher_context=teacher.encode_context(states,candidates[selected]);teacher_logits=teacher.score_candidates(teacher_context,candidates)
 model=SuccessorAffordanceResidual(candidate_size=candidates.shape[1],slot_feature_size=sf.shape[2],hidden_size=cfg["hidden_size"],slot_layers=protocol["stage_c1"]["affordance_builder"]["message_layers"],dropout=cfg["dropout"]).to(device)
 optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"]);history=[]
 discounts={h:extra["successor_discount"]**(h-2) for h in range(2,6)};discount_total=sum(discounts.values())
 for epoch in range(cfg["epochs"]):
  v6.seed(training_seed*4001+epoch);model.train();initial,_=model.initial_hidden(teacher_context,sf,st,sr,sm)
  surface=surfaces[1];keep=np.asarray([events[i]["split"]=="training" for i in surface["starts"]]);index=torch.tensor(surface["starts"][keep],device=device);target=torch.tensor(surface["targets"][keep],device=device);legal=torch.tensor(surface["legals"][keep,-1],dtype=torch.bool,device=device);base=teacher_logits[index].masked_fill(~legal,torch.finfo(torch.float32).min);logits=(teacher_logits[index]+model.one_step_delta_logits(initial[index],candidates)).masked_fill(~legal,torch.finfo(torch.float32).min);weights=torch.tensor(v5._task_weights([events[i] for i in surface["starts"][keep]]),device=device);h1_ce=(F.cross_entropy(logits,target,reduction="none")*weights).sum()/weights.sum();bp=F.softmax(base,1);h1_kl=(bp*(F.log_softmax(base,1)-F.log_softmax(logits,1))).sum(1);h1_kl=(h1_kl*weights).sum()/weights.sum();total=cfg["h1_ce_weight"]*h1_ce+cfg["h1_kl_weight"]*h1_kl;parts={"h1_ce":h1_ce,"h1_kl":h1_kl};successor_total=torch.zeros((),device=device)
  for horizon in range(2,6):
   surface=surfaces[horizon];keep=np.asarray([events[i]["split"]=="training" for i in surface["starts"]]);starts=torch.tensor(surface["starts"][keep],device=device);paths=torch.tensor(surface["paths"][keep],device=device);hidden=initial[starts]
   for step in range(1,horizon):hidden=model.advance(hidden,candidates[paths[:,step]])
   legal=torch.tensor(surface["legals"][keep,-1],dtype=torch.bool,device=device);target=torch.tensor(surface["targets"][keep],device=device);logits=model.rollout_logits(hidden,candidates).masked_fill(~legal,torch.finfo(torch.float32).min);weights=torch.tensor(v5._task_weights([events[i] for i in surface["starts"][keep]]),device=device);hce=(F.cross_entropy(logits,target,reduction="none")*weights).sum()/weights.sum();future=torch.tensor(surface["future"][keep],device=device);latent=1-F.cosine_similarity(model.projected_context(hidden),teacher_context[future],dim=1);latent=(latent*weights).sum()/weights.sum();trainable=np.asarray([events[i]["joint_outcome_trainable"] for i in surface["starts"][keep]]);ji=torch.tensor(np.flatnonzero(trainable),device=device);joint=torch.zeros((),device=device)
   if len(ji):
    y=torch.tensor(np.stack([[events[surface["starts"][keep][i]]["joint_outcome_target"][name] for name in JOINT_OUTCOME_CLASSES] for i in np.flatnonzero(trainable)]),dtype=torch.float32,device=device);joint=-(y*F.log_softmax(model.joint_logits(hidden[ji]),1)).sum(1).mean()
   successor_logits=model.successor_logits(initial[starts],candidates).masked_fill(~legal,torch.finfo(torch.float32).min);successor=(F.cross_entropy(successor_logits,target,reduction="none")*weights).sum()/weights.sum();successor_total=successor_total+discounts[horizon]*successor/discount_total
   total=total+cfg["horizon_weights"][str(horizon)]*hce+cfg["latent_weight"]*latent+cfg["future_joint_weight"]*joint;parts[f"h{horizon}_ce"]=hce;parts[f"h{horizon}_successor"]=successor
  total=total+extra["successor_weight"]*successor_total;parts["successor_total"]=successor_total
  optimizer.zero_grad(set_to_none=True);total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step()
  if epoch in (0,cfg["epochs"]-1):history.append({"epoch":epoch,"total":float(total.detach()),"slot_gate":float(torch.tanh(model.slot_gate).detach()),**{k:float(v.detach()) for k,v in parts.items()}})
 return model,teacher_context,teacher_logits,(sf,st,sr,sm),history
def main():
 p=argparse.ArgumentParser();p.add_argument("--protocol",type=Path,required=True);p.add_argument("--dataset",type=Path,required=True);p.add_argument("--audit",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args();protocol=json.loads(a.protocol.read_text());data=json.loads(a.dataset.read_text())
 if protocol["status"]!="preregistered_stage_c2_before_training":raise ValueError("C2 protocol not frozen")
 if file_sha256(a.dataset)!=protocol["frozen_dataset"]["sha256"] or file_sha256(a.audit)!=protocol["frozen_dataset"]["audit_sha256"]:raise ValueError("frozen data mismatch")
 torch.set_num_threads(8);device="cpu";a.output_dir.mkdir(parents=True,exist_ok=True);pred=a.output_dir/"predictions.jsonl";pred.write_text("");runs=[];all_audits=[];builder=protocol["stage_c1"]["affordance_builder"];teacher_protocol={"training":protocol["teacher"]}
 for fold in range(protocol["research_budget"]["folds"]):
  events=v5._fold(data,fold);arrays=v5._arrays(events,data["candidate_catalog"],128);surfaces=v6.horizons(events,arrays);slots=stack_interface_affordance_states(events,hash_dimension=builder["hash_dimension"],max_nodes=builder["max_nodes"],max_concepts=builder["max_concepts"]);all_audits.extend(slots["audit"])
  for training_seed in protocol["research_budget"]["seeds"]:
   v6.seed(training_seed);values=v5._train("structured_joint_aux",events,arrays,teacher_protocol,training_seed,device,return_model=True);candidate=train(values[4],events,arrays,surfaces,slots,protocol,training_seed,device);rows=stage_a.evaluate_slot(candidate[0],values[4],candidate[1],candidate[2],candidate[3],events,arrays,surfaces,values[3],fold,training_seed,device,arm="successor_affordance_c2");append(pred,rows);runs.append({"fold":fold,"seed":training_seed,"history":candidate[4],"prediction_rows":len(rows)})
 metrics={"training_units":len(runs),"teacher_fits":len(runs),"successor_affordance_residual_fits":len(runs),"runtime_failures":0,"runs":runs,"slot_audit":{"rows":len(all_audits),"raw_values_encoded":any(r["raw_values_encoded"] for r in all_audits),"interface_only_lexical_encoding":all(r["interface_only_lexical_encoding"] for r in all_audits),"unmatched_text_tokens_encoded":sum(r["unmatched_text_tokens_encoded"] for r in all_audits),"truncated_rows":sum(r["truncated"] for r in all_audits),"concept_truncated_rows":sum(r["concepts_truncated"] for r in all_audits),"maximum_nodes":max(r["node_count"] for r in all_audits)},"predictions_sha256":file_sha256(pred)}
 if len(runs)!=15:raise ValueError("fixed C2 budget incomplete")
 write(a.output_dir/"run_metrics.json",metrics)
if __name__=="__main__":main()
