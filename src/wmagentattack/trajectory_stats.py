"""Summary statistics for raw AgentDojo JSON traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _content_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        item.get("content", "") for item in content if isinstance(item, dict)
    )


def summarize_trace(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    trace = json.loads(path.read_text(encoding="utf-8"))
    messages = trace.get("messages", [])
    assistant_messages = [m for m in messages if m.get("role") == "assistant"]
    tool_messages = [m for m in messages if m.get("role") == "tool"]
    tool_calls = [
        call
        for message in assistant_messages
        for call in (message.get("tool_calls") or [])
    ]
    tool_errors = [m for m in tool_messages if m.get("error")]
    parse_failures = [
        m
        for m in assistant_messages
        if "<function=" in _content_text(m) and not (m.get("tool_calls") or [])
    ]
    guessed_2023 = any(
        "2023-" in json.dumps(call.get("args", {}), ensure_ascii=False)
        for call in tool_calls
    )

    completed = trace.get("utility") is not None and trace.get("security") is not None
    return {
        "suite": trace.get("suite_name"),
        "pipeline": trace.get("pipeline_name"),
        "task": trace.get("user_task_id"),
        "completed": completed,
        "utility": trace.get("utility"),
        "security": trace.get("security"),
        "assistant_turns": len(assistant_messages),
        "tool_calls": len(tool_calls),
        "tool_errors": len(tool_errors),
        "parse_failures": len(parse_failures),
        "guessed_2023": guessed_2023,
        "duration_seconds": float(trace.get("duration", 0.0)),
        "trace_path": str(path.resolve()),
    }


def summarize_traces(paths: list[str | Path]) -> dict[str, Any]:
    tasks = [summarize_trace(path) for path in paths]
    completed_tasks = [task for task in tasks if task["completed"]]
    count = len(completed_tasks)
    total_tool_calls = sum(task["tool_calls"] for task in completed_tasks)
    return {
        "num_traces": len(tasks),
        "num_completed_tasks": count,
        "num_incomplete_tasks": len(tasks) - count,
        "utility_rate": (
            sum(bool(task["utility"]) for task in completed_tasks) / count
            if count
            else 0.0
        ),
        "security_rate": (
            sum(bool(task["security"]) for task in completed_tasks) / count
            if count
            else 0.0
        ),
        "avg_assistant_turns": (
            sum(task["assistant_turns"] for task in completed_tasks) / count
            if count
            else 0.0
        ),
        "avg_tool_calls": total_tool_calls / count if count else 0.0,
        "tool_error_rate": (
            sum(task["tool_errors"] for task in completed_tasks) / total_tool_calls
            if total_tool_calls
            else 0.0
        ),
        "parse_failure_rate": (
            sum(task["parse_failures"] for task in completed_tasks) / total_tool_calls
            if total_tool_calls
            else 0.0
        ),
        "tasks_guessing_2023": sum(
            task["guessed_2023"] for task in completed_tasks
        ),
        "total_duration_seconds": sum(
            task["duration_seconds"] for task in completed_tasks
        ),
        "tasks": tasks,
    }
