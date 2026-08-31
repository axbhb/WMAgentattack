from test_protocol_recovery_adapter import adapter
from wmagentattack.postcall_recovery import PostcallRecoveryLLM


HISTORY=[{'role':'user','content':[]},{'role':'assistant','content':[],'tool_calls':[]},{'role':'tool','content':[],'error':None}]
PROMPTS={'generic':'Generic review','grounded':'Observed zero new tool calls'}


def candidate(completions,arm='grounded'):
    a,r=adapter(completions)
    a.__class__=PostcallRecoveryLLM
    a.reset_postcall(arm,PROMPTS)
    return a,r


def test_posttool_intent_gets_exactly_one_additional_generation():
    a,r=candidate(['I will search again.','<function=lookup>{"query":"x"}</function>'])
    result=a.query('query',r,messages=HISTORY)
    assert len(a.model.events)==2 and a.postcall_corrections==1
    assert result[3][-1]['tool_calls'][0].function=='lookup'
    assert len(result[3])==len(HISTORY)+1
    assert a.postcall_probe['instruction']==PROMPTS['grounded']


def test_no_recursive_correction_or_later_episode_reuse():
    a,r=candidate(['I will search again.']*3)
    a.query('query',r,messages=HISTORY)
    a.query('query',r,messages=HISTORY)
    assert len(a.model.events)==3 and a.postcall_corrections==1
    a.reset_postcall('grounded',PROMPTS)
    assert a.postcall_probe is None and a.postcall_corrections==0


def test_anchor_records_same_trigger_without_extra_generation():
    a,r=candidate(['I will search again.'],'anchor')
    a.query('query',r,messages=HISTORY)
    assert a.postcall_probe['prefix_event_count']==1
    assert len(a.model.events)==1 and not a.postcall_corrections


def test_final_answer_without_local_trigger_is_preserved():
    a,r=candidate(['The answer is 42.'])
    a.query('query',r,messages=HISTORY)
    assert len(a.model.events)==1 and a.postcall_probe is None


def test_first_turn_keeps_only_historical_retry_not_posttool_retry():
    a,r=candidate(['I will search now.']*2)
    a.query('query',r,messages=[HISTORY[0]])
    assert len(a.model.events)==2 and a.postcall_corrections==0 and a.postcall_probe is None


def test_generic_and_grounded_share_trigger_and_budget():
    for arm in ('generic','grounded'):
        a,r=candidate(['<function=bad truncated','Completed.'],arm)
        a.query('query',r,messages=HISTORY)
        assert a.postcall_corrections==1 and len(a.model.events)==2
        assert a.postcall_probe['instruction']==PROMPTS[arm]


def test_valid_call_is_not_replaced_and_outcome_metadata_unused():
    a,r=candidate(['<function=lookup>{"query":"x"}</function>'])
    a.query('query',r,messages=HISTORY,extra_args={'utility':False,'security':True})
    assert not a.postcall_corrections and a.postcall_probe is None
