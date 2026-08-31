import json
from pathlib import Path
from wmagentattack.postcall_eval import make_plan,evaluate


def setup():
    root=Path(__file__).resolve().parents[1]
    p=json.loads((root/'configs/0901_postcall_recovery_v42_v43_protocol.json').read_text())['v42']
    tasks=json.loads((root/'configs/0831_clean_pairing_v37_protocol.json').read_text())['tasks']
    rows=make_plan(tasks,p['seeds'],p['arms'])
    for r in rows:
        candidate=r['arm'] in ('generic','grounded')
        r.update(status='completed',utility=candidate,parsed_tool_calls=1,blocked_network_attempts=0,generation_calls=3 if candidate else 2,extra_retry_generations=int(candidate))
    return tasks,p,rows


def artifacts(row):
    candidate=row['arm'] in ('generic','grounded')
    events=[{'completion':'same','input_tokens':2,'output_tokens':1},{'completion':'I will search now.','input_tokens':3,'output_tokens':4}]
    if candidate:events.append({'completion':'corrected','input_tokens':4,'output_tokens':1})
    return {'utility':row['utility'],'messages':[{'role':'assistant','tool_calls':[{}]},{'role':'tool','content':[]},{'role':'assistant','tool_calls':[]}]}, {'events':events,'first_prompt_ids':[1,2],'postcall_corrections':int(candidate),'postcall_probe':None if row['arm']=='strict' else {'prefix_event_count':2,'completion':'I will search now.'}}


def gate(tasks,p,rows,reader=artifacts):return evaluate(rows,tasks,p['seeds'],p['arms'],p['gate'],reader)


def test_balanced_plan_and_generic_preference():
    t,p,rows=setup();assert len(rows)==240 and len({r['episode_seed'] for r in rows})==60
    assert gate(t,p,rows)['selected_arm']=='generic'


def test_missing_row_and_wrong_seed_invalid():
    t,p,rows=setup();assert not gate(t,p,rows[:-1])['scientific_result']
    rows[0]['episode_seed']+=1;assert not gate(t,p,rows)['scientific_result']


def test_trigger_mismatch_invalid_not_silent_candidate_advantage():
    t,p,rows=setup()
    def changed(r):
        raw,d=artifacts(r)
        if r['arm']=='anchor':d['postcall_probe']={'prefix_event_count':1,'completion':'same'}
        return raw,d
    assert not gate(t,p,rows,changed)['scientific_result']


def test_first_prompt_mismatch_invalid():
    t,p,rows=setup()
    def changed(r):
        raw,d=artifacts(r)
        if r['arm']=='grounded':d['first_prompt_ids']=[3]
        return raw,d
    assert not gate(t,p,rows,changed)['scientific_result']


def test_large_gain_cannot_hide_successful_anchor_regression():
    t,p,rows=setup()
    for r in rows:
        if r['task']==t[0]:r['utility']=r['arm']=='anchor'
    g=gate(t,p,rows);assert g['decision']=='NO_GO_POSTCALL_RECOVERY'
    assert not g['comparisons']['grounded']['checks']['preserve_anchor_successes']


def test_duplicate_postcall_correction_invalid():
    t,p,rows=setup()
    def changed(r):
        raw,d=artifacts(r)
        if r['arm']=='generic':d['postcall_corrections']=2
        return raw,d
    assert not gate(t,p,rows,changed)['scientific_result']


def test_confirmation_stability_overlap_is_required():
    t,p,rows=setup();rows=[r for r in rows if r['arm'] in ('anchor','generic')]
    g=evaluate(rows,t,p['seeds'],['anchor','generic'],{**p['gate'],'minimum_stable_overlap':10},artifacts,{'metrics':{'generic':{'stable_tasks':t[:9]}}})
    assert g['decision']=='NO_GO_POSTCALL_RECOVERY'


def test_no_trigger_requires_unchanged_trajectory_and_outcome():
    t,p,rows=setup()
    for r in rows:r.update(utility=False,generation_calls=2,extra_retry_generations=0)
    def no_trigger(r):
        raw,d=artifacts(r)
        d.update(events=d['events'][:2],postcall_corrections=0,postcall_probe=None)
        return raw,d
    assert gate(t,p,rows,no_trigger)['scientific_result']
    next(r for r in rows if r['arm']=='generic')['utility']=True
    assert not gate(t,p,rows,no_trigger)['scientific_result']
