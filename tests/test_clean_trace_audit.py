from wmagentattack.clean_trace_audit import content_text, describe_trace, summarize_traces


def test_content_handles_nested_agentdojo_blocks():
    assert content_text([{"type": "text", "content": "hello"}, {"text": "world"}]) == "hello\nworld"


def test_malformed_tag_text_is_not_an_executed_tool_call():
    row = {"task": "travel|t", "run_seed": 1, "utility": False}
    raw = {"utility": False, "messages": [{"role": "assistant", "content": "<function=bad>", "tool_calls": []}], "error": None}
    detail = describe_trace(row, raw)
    assert detail["function_tag_text_with_zero_calls"]
    assert detail["parsed_tool_calls"] == 0


def test_tool_call_failure_is_separate_from_zero_call_failure():
    row = {"task": "slack|t", "run_seed": 1, "utility": False}
    raw = {"utility": False, "messages": [{"role": "assistant", "tool_calls": [{"function": {"name": "get_users"}}]}, {"role": "tool", "content": "users"}], "error": None}
    detail = describe_trace(row, raw)
    assert detail["parsed_tool_calls"] == 1
    assert detail["tool_names"] == ["get_users"]
    summary = summarize_traces([detail])
    assert summary["failed_despite_parsed_calls"] == 1
    assert summary["failed_zero_parsed_tool_calls"] == 0


def test_raw_label_disagreement_is_visible():
    detail = describe_trace({"task": "workspace|t", "run_seed": 1, "utility": False}, {"utility": True, "messages": []})
    assert not summarize_traces([detail])["raw_utility_all_agree"]


def test_agentdojo_native_function_call_schema():
    detail = describe_trace({"task": "banking|t", "run_seed": 1, "utility": True},
                            {"utility": True, "messages": [{"role": "assistant", "tool_calls": [{"function": "get_transactions", "args": {}}]}]})
    assert detail["tool_names"] == ["get_transactions"]
