"""Post-hoc descriptive failure diagnostics, never used for fitting or gates."""
from collections import Counter, defaultdict


def content_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(content_text(item) for item in content)
    if isinstance(content, dict):
        return content_text(content.get("content", content.get("text", "")))
    return ""


def describe_trace(result, raw):
    messages = raw["messages"]
    assistants = [m for m in messages if m.get("role") == "assistant"]
    calls = [call for message in assistants for call in (message.get("tool_calls") or [])]
    assistant_text = "\n".join(content_text(m.get("content")) for m in assistants)
    names = []
    for call in calls:
        function = call.get("function")
        if isinstance(function, dict):
            name = function.get("name", "unknown")
        elif isinstance(function, str):
            name = function
        else:
            name = call.get("function_name", call.get("name", "unknown"))
        names.append(str(name))
    return {
        "task": result["task"], "run_seed": result["run_seed"],
        "utility": result["utility"], "raw_utility_agrees": raw.get("utility") == result["utility"],
        "assistant_messages": len(assistants), "parsed_tool_calls": len(calls),
        "tool_response_messages": sum(m.get("role") == "tool" for m in messages),
        "function_tag_text_with_zero_calls": len(calls) == 0 and "<function" in assistant_text.lower(),
        "raw_error": raw.get("error"), "tool_names": names,
    }


def summarize_traces(details):
    suites = defaultdict(list)
    for row in details:
        suites[row["task"].split("|", 1)[0]].append(row)
    by_suite = {}
    for suite, rows in sorted(suites.items()):
        failed = [row for row in rows if not row["utility"]]
        by_suite[suite] = {
            "episodes": len(rows), "successes": sum(row["utility"] for row in rows),
            "failed": len(failed),
            "failed_zero_parsed_tool_calls": sum(row["parsed_tool_calls"] == 0 for row in failed),
            "failed_tag_text_zero_calls": sum(row["function_tag_text_with_zero_calls"] for row in failed),
            "failed_despite_parsed_calls": sum(row["parsed_tool_calls"] > 0 for row in failed),
        }
    failed = [row for row in details if not row["utility"]]
    zero = [row for row in failed if row["parsed_tool_calls"] == 0]
    return {
        "scope": "Post-hoc description only, not causal proof or a modified gate",
        "episodes": len(details), "successes": sum(row["utility"] for row in details),
        "failed": len(failed), "failed_zero_parsed_tool_calls": len(zero),
        "failed_tag_text_zero_calls": sum(row["function_tag_text_with_zero_calls"] for row in zero),
        "failed_no_tag_text_zero_calls": sum(not row["function_tag_text_with_zero_calls"] for row in zero),
        "failed_despite_parsed_calls": len(failed) - len(zero),
        "successful_zero_parsed_tool_calls": sum(row["utility"] and row["parsed_tool_calls"] == 0 for row in details),
        "raw_error_counts": dict(Counter(str(row["raw_error"]) for row in details if row["raw_error"])),
        "raw_utility_all_agree": all(row["raw_utility_agrees"] for row in details),
        "by_suite": by_suite, "details": details,
    }
