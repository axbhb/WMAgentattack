"""Frozen remote clean-only post-call feedback comparison; one model load."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from wmagentattack.clean_pairing import block_python_network
from wmagentattack.postcall_eval import make_plan,evaluate


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');tmp.replace(path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--stage',choices=['v42','v43'],required=True);args=ap.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Remote Slurm allocation required')
    p=json.loads(args.protocol.read_text());parent=json.loads(Path(p['parent_protocol']).read_text());stage=p[args.stage];root=Path(stage['archive'])
    audit=json.loads((Path(p['audit_archive'])/'audit.json').read_text());assert audit['decision']=='DESCRIPTIVE_AUDIT_COMPLETE_V41'
    prior=None
    if args.stage=='v43':
        prior=json.loads((Path(p['v42']['archive'])/'gate.json').read_text())
        if prior['decision']!='GO_POSTCALL_RECOVERY' or not prior['selected_arm']:raise RuntimeError('No passing v42 selection')
        arms=['anchor',prior['selected_arm']]
    else:arms=stage['arms']
    root.mkdir(parents=True,exist_ok=True)
    with (root/'execution.lock').open('x') as f:f.write(os.environ['SLURM_JOB_ID'])
    plan=make_plan(parent['tasks'],stage['seeds'],arms);assert len(plan)==stage['episodes']
    write(root/'run_plan.json',plan);write(root/'effective_contract.json',{'model':parent['model'],'tasks':parent['tasks'],'arms':arms,'stage':args.stage})
    os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HUB_OFFLINE']='1'
    import numpy as np
    import torch
    assigned,visible=os.environ.get('SLURM_JOB_GPUS'),os.environ.get('CUDA_VISIBLE_DEVICES')
    init=int(ctypes.CDLL('libcuda.so.1').cuInit(0));smi=subprocess.run(['nvidia-smi'],capture_output=True,text=True)
    preflight={'assigned_gpu':assigned,'visible_gpu':visible,'cuInit':init,'torch_cuda':torch.cuda.is_available(),'device_count':torch.cuda.device_count(),'nvidia_smi_exit':smi.returncode,'nvidia_smi':smi.stdout}
    preflight['passed']=bool(assigned and assigned==visible and init==0 and preflight['torch_cuda'] and preflight['device_count']==1 and smi.returncode==0)
    write(root/'cuda_preflight.json',preflight)
    if not preflight['passed']:raise RuntimeError('CUDA preflight failed before model loading')
    from agentdojo.agent_pipeline import AgentPipeline
    from agentdojo.agent_pipeline.agent_pipeline import PipelineConfig
    from agentdojo.benchmark import run_task_without_injection_tasks
    from agentdojo.logging import OutputLogger
    from agentdojo.task_suite.load_suites import get_suite
    from wmagentattack.postcall_recovery import PostcallRecoveryLLM
    from wmagentattack.protocol_recovery_adapter import GenerationRecorder
    m=parent['model'];rows=[]
    with block_python_network() as network:
        llm=PostcallRecoveryLLM(m['path'],device='cuda:0',quantization=m['quantization'],model_label=f'meta-llama-3.1-70b-{args.stage}',seed=stage['seeds'][0],max_new_tokens=m['max_new_tokens'],max_input_tokens=m['max_input_tokens'],max_tool_output_chars=m['max_tool_output_chars'],prompt_profile=m['prompt_profile'],protocol='function_tags',do_sample=m['do_sample'],temperature=m['temperature'],top_p=m['top_p'])
        llm.model=GenerationRecorder(llm.model,llm.tokenizer)
        pipeline=AgentPipeline.from_config(PipelineConfig(llm=llm,model_id=None,defense=None,system_message_name=None,system_message=None,tool_output_format=None))
        for index,item in enumerate(plan):
            suite_name,task_id=item['task'].split('|',1);seed=item['episode_seed']
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            llm.seed=seed;llm.protocol='function_tags' if item['arm']=='strict' else 'function_tags_repair_retry'
            llm.recovered_calls=0;llm.model.reset();llm.reset_postcall(item['arm'],p['prompts'])
            pipeline.name=f"meta-llama-3.1-70b-{args.stage}-{item['arm']}-local"
            logdir=root/'raw'/item['arm']/f"seed{item['run_seed']}";logdir.mkdir(parents=True,exist_ok=True)
            start=time.monotonic()
            try:
                suite=get_suite('v1.2.2',suite_name)
                with OutputLogger(str(logdir)):
                    utility,_=run_task_without_injection_tasks(suite,pipeline,suite.get_user_task_by_id(task_id),logdir,False,'v1.2.2')
                rawpath=logdir/pipeline.name/suite_name/task_id/'none'/'none.json';raw=json.loads(rawpath.read_text())
                if network['blocked_attempts']:raise RuntimeError('Network attempt in sandbox')
                diagpath=root/'diagnostics'/f'episode_{index:03d}.json'
                d={'first_prompt_ids':llm.model.first_prompt_ids,'events':llm.model.events,'recovered_calls':llm.recovered_calls,'postcall_probe':llm.postcall_probe,'postcall_corrections':llm.postcall_corrections}
                write(diagpath,d)
                extra=len(d['events'])-sum(msg['role']=='assistant' for msg in raw['messages'])
                if extra-llm.postcall_corrections not in ([0] if item['arm']=='strict' else [0,1]):raise RuntimeError('Correction generation budget mismatch')
                row={**item,'status':'completed','utility':bool(utility),'raw_trace':str(rawpath),'diagnostics':str(diagpath),'blocked_network_attempts':network['blocked_attempts'],'parsed_tool_calls':sum(len(msg.get('tool_calls') or []) for msg in raw['messages'] if msg['role']=='assistant'),'generation_calls':len(d['events']),'extra_retry_generations':extra,'postcall_corrections':llm.postcall_corrections,'elapsed_seconds':time.monotonic()-start}
            except Exception as error:
                row={**item,'status':'failed','error_type':type(error).__name__,'error':str(error)};rows.append(row);write(root/'results.json',rows);write(root/'gate.json',{'decision':'INVALID_POSTCALL_RECOVERY','scientific_result':False,'failure':row});raise
            rows.append(row);write(root/'results.json',rows);print(json.dumps({'recorded':len(rows),'expected':len(plan),**row}),flush=True)
    def artifacts(row):
        a,b=Path(row['raw_trace']).resolve(),Path(row['diagnostics']).resolve()
        if not a.is_relative_to(root/'raw') or not b.is_relative_to(root/'diagnostics'):raise RuntimeError('Artifact outside archive')
        return json.loads(a.read_text()),json.loads(b.read_text())
    gate=evaluate(rows,parent['tasks'],stage['seeds'],arms,stage['gate'],artifacts,prior)
    gate.update(stage=args.stage,slurm_job_id=os.environ['SLURM_JOB_ID']);write(root/'gate.json',gate)
    if not gate['scientific_result']:raise RuntimeError('Final integrity gate failed')
    print(json.dumps(gate),flush=True)


if __name__=='__main__':main()
