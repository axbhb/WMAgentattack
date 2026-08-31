"""Observable trace indicators; intentionally independent of utility/security labels."""
import json
import re


def text_of(message):
    content = message.get("content") or []
    if isinstance(content, str):
        return content
    return "\n".join(c.get("content", "") for c in content if c.get("type") == "text")


def indicators(messages, events, should_retry_intent):
    tools = [m for m in messages if m["role"] == "tool"]
    assistants = [m for m in messages if m["role"] == "assistant"]
    terminal = assistants[-1] if assistants else {}
    terminal_text = text_of(terminal)
    calls = [c for m in assistants for c in (m.get("tool_calls") or [])]
    signatures = [json.dumps({"function": c["function"], "args": c["args"]}, sort_keys=True) for c in calls]
    no_terminal_call = bool(assistants) and not terminal.get("tool_calls")
    return {
        "tool_error_count": sum(bool(m.get("error")) for m in tools),
        "terminal_after_tool": bool(tools) and no_terminal_call,
        "terminal_unparsed_marker_after_tool": bool(tools) and no_terminal_call and bool(re.search(r"<function\b", terminal_text, re.I)),
        "terminal_intent_after_tool": bool(tools) and no_terminal_call and should_retry_intent(terminal_text),
        "consecutive_repeated_call_count": sum(a == b for a,b in zip(signatures, signatures[1:])),
        "input_cap_events": sum(e["input_tokens"] >= 8192 for e in events),
        "output_cap_events": sum(e["output_tokens"] >= 256 for e in events),
        "parsed_calls": len(calls),
    }
