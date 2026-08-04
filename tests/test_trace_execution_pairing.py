from wmagentattack.trace_execution_pairing import pair_executed_clean_tool_calls


def _assistant(function=None, args=None, text=""):
    calls = []
    if function is not None:
        calls.append({"function": function, "args": args or {}})
    return {
        "role": "assistant",
        "content": [{"type": "text", "content": text}],
        "tool_calls": calls,
    }


def _tool(function, args=None, error=None):
    return {
        "role": "tool",
        "content": [{"type": "text", "content": "not retained"}],
        "tool_call": {"function": function, "args": args or {}},
        "error": error,
    }


def test_terminal_unexecuted_proposal_is_not_an_executed_pair():
    pairing, final_output = pair_executed_clean_tool_calls(
        [
            _assistant("search", {"city": "Paris"}),
            _tool("search", {"city": "Paris"}),
            _assistant("search", {"city": "London"}, text="terminal proposal"),
        ]
    )
    assert pairing.proposal_count == 2
    assert pairing.tool_message_count == 1
    assert len(pairing.executed_pairs) == 1
    assert pairing.executed_pairs[0].proposal.arguments == {"city": "Paris"}
    assert len(pairing.terminal_unexecuted_proposals) == 1
    assert pairing.terminal_unexecuted_proposals[0].arguments == {"city": "London"}
    assert pairing.executed_alignment_ok
    assert final_output == "terminal proposal"


def test_displaced_proposal_is_midtrajectory_defect():
    pairing, _ = pair_executed_clean_tool_calls(
        [
            _assistant("search", {"city": "Paris"}),
            _assistant(None, text="final"),
        ]
    )
    assert len(pairing.midtrajectory_unexecuted_proposals) == 1
    assert not pairing.executed_alignment_ok


def test_error_flag_is_paired_without_tool_output_or_outcome():
    pairing, _ = pair_executed_clean_tool_calls(
        [
            _assistant("search", {"city": "Paris"}),
            _tool("search", {"city": "Paris"}, error="ValidationError"),
            _assistant(None, text="done"),
        ]
    )
    assert pairing.executed_pairs[0].logged_error is True
    payload = pairing.model_dump(mode="json")
    assert "content" not in str(payload)
    assert payload["outcome_labels_present"] is False
    assert pairing.executed_alignment_ok
