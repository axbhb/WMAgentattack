import json

from wmagentattack.trajectory_stats import summarize_trace, summarize_traces


def test_summarize_trace_counts_calls_and_errors(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(
        json.dumps(
            {
                "suite_name": "workspace",
                "user_task_id": "user_task_0",
                "utility": True,
                "security": True,
                "duration": 2.5,
                "messages": [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "content": "<function=x>{}</function>"}],
                        "tool_calls": [{"function": "x", "args": {"date": "2023-01-01"}}],
                    },
                    {"role": "tool", "content": [], "error": "bad input"},
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "content": "done"}],
                        "tool_calls": [],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = summarize_trace(path)
    assert result["tool_calls"] == 1
    assert result["tool_errors"] == 1
    assert result["parse_failures"] == 0
    assert result["guessed_2023"] is True
    assert result["completed"] is True

    combined = summarize_traces([path])
    assert combined["utility_rate"] == 1.0
    assert combined["tool_error_rate"] == 1.0
