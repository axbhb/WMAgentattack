"""Apply the complete frozen gate to the tau3 tail-pilot base summary."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.tau3_multistep import file_sha256
from wmagentattack.tau3_tail_horizon import evaluate_tail_gate


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _episodes(directory: Path) -> list[dict[str, Any]]:
    result = []
    for path in sorted(directory.glob("chunk*.json")):
        result.extend(json.loads(path.read_text(encoding="utf-8"))["episodes"])
    return result


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# tau3 bounded-tail horizon pilot",
        "",
        f"Decision: `{summary['decision']}`",
        "",
        "| Metric | Result |",
        "|---|---:|",
    ]
    for key, value in summary["metrics"].items():
        rendered = f"{value:.6f}" if isinstance(value, float) else str(value)
        lines.append(f"| {key} | {rendered} |")
    lines.extend(["", "## Frozen gate", ""])
    lines.extend(f"- {key}: `{'PASS' if value else 'FAIL'}`" for key, value in summary["gate_checks"].items())
    lines.extend([
        "",
        "## Claim boundary",
        "",
        "A GO authorizes only a separately frozen 96-episode 20/20/80 confirmation. It does not authorize attacks, Dreamer training, or unrestricted tau3 scale-up.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--base-gate", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    base = json.loads(args.base_gate.read_text(encoding="utf-8"))
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    episodes = _episodes(args.output_dir)
    metrics = dict(base["metrics"])
    assistant_events = [event for episode in episodes for event in episode["combined_tool_events"] if event["requestor"] == "assistant"]
    per_episode = Counter()
    tasks = set()
    tools = set()
    communication_errors = 0
    natural_messages = 0
    for episode in episodes:
        events = [event for event in episode["combined_tool_events"] if event["requestor"] == "assistant"]
        per_episode[str(episode["episode_id"])] = len(events)
        if events:
            tasks.add(str(episode["task_key"]))
            tools.update(str(event["action"]["name"]) for event in events)
        termination = str(episode["termination"]).lower()
        communication_errors += int(termination in {"agent_error", "user_error"} or termination.endswith(".agent_error") or termination.endswith(".user_error"))
        natural_messages += int(episode["natural_user_message_count"])
    action_counts = Counter(
        str(prefix["targets"]["next_action"])
        for episode in dataset["episodes"]
        for prefix in episode["prefixes"]
    )
    decisions = sum(action_counts.values())
    tool_decisions = sum(value for key, value in action_counts.items() if not key.endswith("::TEXT"))
    metrics.update({
        "communication_error_terminations": communication_errors,
        "natural_user_messages": natural_messages,
        "episodes_with_two_or_more_assistant_transitions": sum(value >= 2 for value in per_episode.values()),
        "tasks_with_at_least_one_assistant_transition": len(tasks),
        "unique_executed_assistant_tools": len(tools),
        "agent_tool_decision_rate": tool_decisions / max(decisions, 1),
        "dominant_agent_action_fraction": max(action_counts.values(), default=0) / max(decisions, 1),
    })
    integrity = dict(base["integrity_checks"])
    integrity["complete_chunk_surface"] = len(list(args.output_dir.glob("chunk*.json"))) == int(protocol["execution"]["summary_chunks_expected"])
    checks = evaluate_tail_gate(metrics, integrity, protocol["pilot_gate"])
    passed = all(checks.values())
    summary = {
        "protocol_id": protocol["protocol_id"],
        "decision": "TAIL_HORIZON_PILOT_GO__AUTHORIZE_96_CONFIRMATION" if passed else "TAIL_HORIZON_PILOT_NO_GO__DO_NOT_CONFIRM_OR_SCALE",
        "passed": passed,
        "metrics": metrics,
        "gate_checks": checks,
        "integrity_checks": integrity,
        "base_gate_sha256": file_sha256(args.base_gate),
        "dataset_sha256": file_sha256(args.dataset),
        "protocol_sha256": file_sha256(args.protocol),
    }
    _write(args.output, summary)
    args.report.write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
