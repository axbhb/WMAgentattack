from wmagentattack.postcall_audit import indicators


def features(messages, events=()):
    return indicators(messages, events, lambda s: s.startswith("I will search"))


def test_marker_is_only_posttool_and_unparsed():
    m={"role":"assistant","content":"<function=lookup bad>","tool_calls":[]}
    assert not features([m])["terminal_unparsed_marker_after_tool"]
    assert features([{"role":"tool","error":None},m])["terminal_unparsed_marker_after_tool"]
    assert not features([{"role":"tool","error":None},{**m,"tool_calls":[{"function":"lookup","args":{}}]}])["terminal_unparsed_marker_after_tool"]


def test_error_requires_explicit_error_field_not_word_in_content():
    assert features([{"role":"tool","content":"error discussed","error":None}])["tool_error_count"]==0
    assert features([{"role":"tool","error":"Invalid arguments"}])["tool_error_count"]==1


def test_repetition_and_caps_are_indicators_not_success_labels():
    m={"role":"assistant","tool_calls":[{"function":"lookup","args":{"x":1}}]}
    f=features([m,m],[{"input_tokens":8192,"output_tokens":256}])
    assert f["consecutive_repeated_call_count"]==1 and f["input_cap_events"]==f["output_cap_events"]==1


def test_intent_and_ordinary_answer_distinct():
    history=[{"role":"tool","error":None}]
    assert features(history+[{"role":"assistant","content":"I will search again"}])["terminal_intent_after_tool"]
    assert not features(history+[{"role":"assistant","content":"Completed."}])["terminal_intent_after_tool"]
