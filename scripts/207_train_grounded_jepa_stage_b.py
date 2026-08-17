"""Stage B: action-conditioned JEPA and semantic grounding for relational slots."""
from __future__ import annotations
import argparse,copy,importlib.util,json,random,sys
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from wmagentattack.joint_outcome_auxiliary import JOINT_OUTCOME_CLASSES
from wmagentattack.multisource_suitability import file_sha256
from wmagentattack.relational_slot_latent import GroundedPredictiveSlotResidual,stack_relational_slot_states

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
v5=load("v5",ROOT/"scripts/201_train_structured_joint_outcome_v5.py");v6=load("v6",ROOT/"scripts/203_train_structured_residual_v6.py");stage_a=load("stage_a",ROOT/"scripts/205_train_relational_slot_stage_a.py")
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
def append(path,rows):
 with path.open("a") as f:
  for row in rows:f.write(json.dumps(row,sort_keys=True)+"\n")
def restore(state):random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2])
def vicreg(latent):
 centered=latent-latent.mean(0,keepdim=True);std=torch.sqrt(centered.var(0,unbiased=False)+1e-4);variance=F.relu(1-std).mean();cov=centered.T@centered/max(1,len(latent)-1);off=cov-torch.diag(torch.diag(cov));return variance,(off.square().sum()/latent.shape[1]),std
def train_grounded(teacher,events,arrays,surfaces,slots,protocol,training_seed,device):
 cfg=protocol["stage_a"]["residual"];extra=protocol["stage_b"]["training"];teacher.eval()
 for parameter in teacher.parameters():parameter.requires_grad_(False)
 states=torch.tensor(arrays["states"],dtype=torch.float32,device=device);candidates=torch.tensor(arrays["candidate_inputs"],dtype=torch.float32,device=device);selected=torch.tensor(arrays["selected"],dtype=torch.long,device=device)
 sf=torch.tensor(slots["features"],dtype=torch.float32,device=device);st=torch.tensor(slots["node_types"],dtype=torch.long,device=device);sr=torch.tensor(slots["relations"],dtype=torch.long,device=device);sm=torch.tensor(slots["mask"],dtype=torch.bool,device=device);ground=torch.tensor(slots["grounding"],dtype=torch.float32,device=device)
 with torch.no_grad():teacher_context=teacher.encode_context(states,candidates[selected]);teacher_logits=teacher.score_candidates(teacher_context,candidates)
 model=GroundedPredictiveSlotResidual(candidate_size=candidates.shape[1],slot_feature_size=sf.shape[2],hidden_size=cfg["hidden_size"],slot_layers=protocol["stage_a"]["slot_builder"]["message_layers"],grounding_size=ground.shape[1],dropout=cfg["dropout"]).to(device)
 target_encoder=copy.deepcopy(model.slot_encoder).to(device).eval()
 for parameter in target_encoder.parameters():parameter.requires_grad_(False)
 optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"]);history=[];train_rows=torch.tensor([i for i,e in enumerate(events) if e["split"]=="training"],device=device)
 for epoch in range(cfg["epochs"]):
  v6.seed(training_seed*3001+epoch);model.train();initial,slot=model.initial_hidden(teacher_context,sf,st,sr,sm)
  static=F.mse_loss(model.static_grounding(slot[train_rows]),ground[train_rows]);variance,covariance,std=vicreg(slot[train_rows])
  surface=surfaces[1];keep=np.asarray([events[i]["split"]=="training" for i in surface["starts"]]);index=torch.tensor(surface["starts"][keep],device=device);target=torch.tensor(surface["targets"][keep],device=device);legal=torch.tensor(surface["legals"][keep,-1],dtype=torch.bool,device=device)
  base=teacher_logits[index].masked_fill(~legal,torch.finfo(torch.float32).min);logits=(teacher_logits[index]+model.one_step_delta_logits(initial[index],candidates)).masked_fill(~legal,torch.finfo(torch.float32).min);weights=torch.tensor(v5._task_weights([events[i] for i in surface["starts"][keep]]),device=device)
  h1_ce=(F.cross_entropy(logits,target,reduction="none")*weights).sum()/weights.sum();bp=F.softmax(base,1);h1_kl=(bp*(F.log_softmax(base,1)-F.log_softmax(logits,1))).sum(1);h1_kl=(h1_kl*weights).sum()/weights.sum()
  total=cfg["h1_ce_weight"]*h1_ce+cfg["h1_kl_weight"]*h1_kl+extra["static_grounding_weight"]*static+extra["variance_weight"]*variance+extra["covariance_weight"]*covariance;parts={"h1_ce":h1_ce,"h1_kl":h1_kl,"static":static,"variance":variance,"covariance":covariance,"slot_std":std.mean()}
  for horizon in range(2,6):
   surface=surfaces[horizon];keep=np.asarray([events[i]["split"]=="training" for i in surface["starts"]]);starts=torch.tensor(surface["starts"][keep],device=device);paths=torch.tensor(surface["paths"][keep],device=device);hidden=initial[starts]
   for step in range(1,horizon):hidden=model.advance(hidden,candidates[paths[:,step]])
   legal=torch.tensor(surface["legals"][keep,-1],dtype=torch.bool,device=device);target=torch.tensor(surface["targets"][keep],device=device);logits=model.rollout_logits(hidden,candidates).masked_fill(~legal,torch.finfo(torch.float32).min);weights=torch.tensor(v5._task_weights([events[i] for i in surface["starts"][keep]]),device=device);hce=(F.cross_entropy(logits,target,reduction="none")*weights).sum()/weights.sum();future=torch.tensor(surface["future"][keep],device=device)
   teacher_latent=(1-F.cosine_similarity(model.projected_context(hidden),teacher_context[future],dim=1));teacher_latent=(teacher_latent*weights).sum()/weights.sum();predicted_slot=model.predict_slot_latent(hidden)
   with torch.no_grad():target_slot=target_encoder(sf[future],st[future],sr[future],sm[future])
   jepa=1-F.cosine_similarity(F.normalize(predicted_slot,dim=1),F.normalize(target_slot,dim=1),dim=1);jepa=(jepa*weights).sum()/weights.sum();delta_target=ground[future]-ground[starts];transition=F.mse_loss(model.transition_grounding(slot[starts],predicted_slot),delta_target)
   trainable=np.asarray([events[i]["joint_outcome_trainable"] for i in surface["starts"][keep]]);ji=torch.tensor(np.flatnonzero(trainable),device=device);joint=torch.zeros((),device=device)
   if len(ji):
    y=torch.tensor(np.stack([[events[surface["starts"][keep][i]]["joint_outcome_target"][name] for name in JOINT_OUTCOME_CLASSES] for i in np.flatnonzero(trainable)]),dtype=torch.float32,device=device);joint=-(y*F.log_softmax(model.joint_logits(hidden[ji]),1)).sum(1).mean()
   total=total+cfg["horizon_weights"][str(horizon)]*hce+cfg["latent_weight"]*teacher_latent+cfg["future_joint_weight"]*joint+extra["jepa_weight"]*jepa+extra["transition_grounding_weight"]*transition;parts[f"h{horizon}_ce"]=hce;parts[f"h{horizon}_jepa"]=jepa;parts[f"h{horizon}_transition"]=transition
  optimizer.zero_grad(set_to_none=True);total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step()
  with torch.no_grad():
   for target_parameter,online_parameter in zip(target_encoder.parameters(),model.slot_encoder.parameters()):target_parameter.mul_(extra["ema_decay"]).add_(online_parameter,alpha=1-extra["ema_decay"])
  if epoch in (0,cfg["epochs"]-1):history.append({"epoch":epoch,"total":float(total.detach()),"slot_gate":float(torch.tanh(model.slot_gate).detach()),**{key:float(value.detach()) for key,value in parts.items()}})
 model.eval()
 with torch.no_grad():
  _,confirmation_slot=model.initial_hidden(teacher_context,sf,st,sr,sm);confirmation=confirmation_slot[torch.tensor([i for i,e in enumerate(events) if e["split"]=="confirmation"],device=device)];_,_,confirmation_std=vicreg(confirmation)
 diagnostics={"slot_std_mean":float(confirmation_std.mean()),"low_variance_fraction":float((confirmation_std<0.05).float().mean())}
 return model,teacher_context,teacher_logits,(sf,st,sr,sm),history,diagnostics
def main():
 p=argparse.ArgumentParser();p.add_argument("--protocol",type=Path,required=True);p.add_argument("--dataset",type=Path,required=True);p.add_argument("--audit",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);args=p.parse_args();protocol=json.loads(args.protocol.read_text());data=json.loads(args.dataset.read_text())
 if file_sha256(args.dataset)!=protocol["frozen_dataset"]["sha256"] or file_sha256(args.audit)!=protocol["frozen_dataset"]["audit_sha256"]:raise ValueError("frozen data mismatch")
 torch.set_num_threads(8);device="cpu";args.output_dir.mkdir(parents=True,exist_ok=True);pred=args.output_dir/"predictions.jsonl";pred.write_text("");runs=[];teacher_protocol={"training":protocol["stage_a"]["teacher"]}
 for fold in range(5):
  events=v5._fold(data,fold);arrays=v5._arrays(events,data["candidate_catalog"],128);surfaces=v6.horizons(events,arrays);slots=stack_relational_slot_states(events,hash_dimension=protocol["stage_a"]["slot_builder"]["hash_dimension"],max_nodes=protocol["stage_a"]["slot_builder"]["max_nodes"])
  for training_seed in protocol["research_budget"]["seeds"]:
   v6.seed(training_seed);values=v5._train("structured_joint_aux",events,arrays,teacher_protocol,training_seed,device,return_model=True);teacher=values[4];rng=(random.getstate(),np.random.get_state(),torch.get_rng_state())
   baseline,context,logits,baseline_history=v6.train_residual(teacher,events,arrays,surfaces,{"residual_training":protocol["stage_a"]["residual"]},training_seed,device);baseline_rows=[row for row in v6.evaluate(baseline,teacher,context,logits,events,arrays,surfaces,values[3],fold,training_seed,device) if row["arm"]=="structured_residual_v6"]
   for row in baseline_rows:row["arm"]="v6_replication_stage_b"
   append(pred,baseline_rows);restore(rng);candidate=train_grounded(teacher,events,arrays,surfaces,slots,protocol,training_seed,device);candidate_rows=stage_a.evaluate_slot(candidate[0],teacher,candidate[1],candidate[2],candidate[3],events,arrays,surfaces,values[3],fold,training_seed,device,arm="grounded_jepa_stage_b");append(pred,candidate_rows);runs.append({"fold":fold,"seed":training_seed,"v6_history":baseline_history,"jepa_history":candidate[4],**candidate[5]})
 metrics={"training_units":len(runs),"teacher_fits":len(runs),"v6_residual_fits":len(runs),"grounded_jepa_fits":len(runs),"runtime_failures":0,"runs":runs,"predictions_sha256":file_sha256(pred)}
 if len(runs)!=15:raise ValueError("budget incomplete")
 write(args.output_dir/"run_metrics.json",metrics)
if __name__=="__main__":main()
