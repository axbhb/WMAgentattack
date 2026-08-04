import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "120_replay_observed_clean_state.py"
SPEC = importlib.util.spec_from_file_location("observed_clean_state_replay", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_clean_contract_rejects_any_injection_metadata():
    clean = {
        "attack_type": "none",
        "injection_task_id": None,
        "injections": {},
        "messages": [],
    }
    assert MODULE._clean_contract(clean) == (True, ())
    attacked = {**clean, "injection_task_id": "injection_task_1"}
    ok, failures = MODULE._clean_contract(attacked)
    assert not ok
    assert failures == ("injection_task_id",)


def test_call_extraction_alignment_and_text_separation():
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "text", "content": "calling"}],
            "tool_calls": [{"function": "search", "args": {"city": "Paris"}}],
        },
        {
            "role": "tool",
            "content": [{"type": "text", "content": "raw output"}],
            "tool_call": {"function": "search", "args": {"city": "Paris"}},
            "error": None,
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "content": "final answer"}],
            "tool_calls": [],
        },
    ]
    calls, tools, output, multi = MODULE._extract_calls_and_output(messages)
    assert calls == [{"function": "search", "args": {"city": "Paris"}}]
    assert tools[0]["has_error"] is False
    assert output == "final answer"
    assert multi == 0
    assert MODULE._calls_align(calls, tools)
    tools[0]["args"] = {"city": "London"}
    assert not MODULE._calls_align(calls, tools)


def test_mixed_utility_groups_require_same_frozen_key():
    rows = [
        {
            "user_task_id": "task",
            "final_state_fingerprint": "state-a",
            "archived_utility": False,
            "panel": "development",
            "seed": 1,
        },
        {
            "user_task_id": "task",
            "final_state_fingerprint": "state-a",
            "archived_utility": True,
            "panel": "confirmation",
            "seed": 2,
        },
        {
            "user_task_id": "task",
            "final_state_fingerprint": "state-b",
            "archived_utility": True,
            "panel": "confirmation",
            "seed": 3,
        },
    ]
    groups = MODULE._mixed_groups(
        rows, ("user_task_id", "final_state_fingerprint")
    )
    assert len(groups) == 1
    assert groups[0]["episodes"] == 2
    assert groups[0]["successes"] == 1
