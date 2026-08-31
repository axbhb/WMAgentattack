"""Frozen v41 remote diagnostic, never executes a model or task."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from wmagentattack.postcall_audit import indicators, text_of
from wmagentattack.qwen_agentdojo import TransformersQwenLLM
from wmagentattack.clean_pairing import block_python_network


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--protocol",type=Path,required=True);args=ap.parse_args()
    p=json.loads(args.protocol.read_text());source=Path(p["source_archive"]);out=Path(p["archive"])
    rows=json.loads((source/"results.json").read_text());assert len(rows)==180
    assert len({(r['arm'],r['task'],r['run_seed']) for r in rows})==180
    details=[]
    with block_python_network() as network:
        for r in rows:
            assert r['status']=='completed'
            raw=json.loads(Path(r['raw_trace']).read_text());diag=json.loads(Path(r['diagnostics']).read_text())
            assert raw['utility']==r['utility'] and not raw.get('error')
            f=indicators(raw['messages'],diag['events'],TransformersQwenLLM._should_retry_tool_intent)
            assert f['parsed_calls']==r['parsed_tool_calls']
            assistants=[m for m in raw['messages'] if m['role']=='assistant']
            details.append({'arm':r['arm'],'task':r['task'],'seed':r['run_seed'],'utility':r['utility'],'indicators':f,
                'terminal_excerpt':text_of(assistants[-1])[:1600] if assistants else '',
                'tool_errors':[{'function':m.get('tool_call',{}).get('function'),'error':m['error']} for m in raw['messages'] if m['role']=='tool' and m.get('error')]})
        assert network['blocked_attempts']==0
    groups={}
    for arm in ['strict','syntax','syntax_retry']:
        for success in [False,True]:
            selected=[r for r in details if r['arm']==arm and r['utility']==success]
            groups[f'{arm}_success_{int(success)}']={'episodes':len(selected),'indicator_episode_counts':dict(Counter(k for r in selected for k,v in r['indicators'].items() if v)),
                'indicator_task_counts':{k:len({r['task'] for r in selected if r['indicators'][k]}) for k in details[0]['indicators']}}
    result={'decision':'DESCRIPTIVE_AUDIT_COMPLETE_V41','source_rows':180,'groups':groups,'details':details,'checked_at_utc':datetime.now(timezone.utc).isoformat(),'new_generations':0,'tool_executions':0}
    with (out/'audit.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'decision':result['decision'],'groups':groups},indent=2))
    print('FAILED_POSTCALL_EXCERPTS')
    for r in details:
        if r['arm']=='syntax_retry' and not r['utility'] and r['indicators']['parsed_calls']:
            print(json.dumps(r))


if __name__=='__main__':main()
