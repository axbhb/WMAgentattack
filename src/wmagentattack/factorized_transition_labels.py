"""Causal, discrete semantic-transition targets for AgentDojo trajectories."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence


OBSERVATION_DELTA_CLASSES = (
    "unchanged", "append_growth", "novel_gain", "rewrite_or_loss",
)
EVIDENCE_DELTA_CLASSES = (
    "none", "goal_gain", "interface_gain", "mixed_gain",
)
GOAL_PROGRESS_CLASSES = (
    "regress", "steady", "advance_small", "advance_large",
)
EXECUTION_STATUS_CLASSES = (
    "error_empty_continue", "productive_continue", "productive_stop",
    "empty_continue", "empty_stop",
)
FACTOR_CLASSES = {
    "observation_delta": OBSERVATION_DELTA_CLASSES,
    "evidence_delta": EVIDENCE_DELTA_CLASSES,
    "goal_progress": GOAL_PROGRESS_CLASSES,
    "execution_status": EXECUTION_STATUS_CLASSES,
}

_TOKEN = re.compile(r"[a-z][a-z0-9_]+", re.I)
_STOP = frozenset(
    "a an and are as at be been by can could did do does for from get given has have "
    "if in into is it its may of on or return that the their them these this to type use "
    "used user using was were when where which will with would information data value values".split()
)


def _tokens(value: object) -> set[str]:
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False)
    output = set()
    for raw in _TOKEN.findall(value.lower()):
        for token in raw.split("_"):
            if len(token) >= 3 and token not in _STOP:
                output.add(token)
    return output


def _schema_tokens(causal: Mapping[str, object]) -> set[str]:
    return _tokens(causal.get("legal_tool_names", ())) | _tokens(causal.get("tool_schemas", ()))


def _observation_delta(current: set[str], following: set[str]) -> str:
    if current == following:
        return "unchanged"
    new = following - current; removed = current - following
    if current and not removed and new:
        return "append_growth"
    union = current | following
    jaccard = len(current & following) / max(len(union), 1)
    if len(new) > len(removed) and jaccard >= 0.4:
        return "novel_gain"
    return "rewrite_or_loss"


def _evidence_delta(
    new_tokens: set[str], goal_tokens: set[str], interface_tokens: set[str]
) -> str:
    goal = bool(new_tokens & goal_tokens)
    interface = bool(new_tokens & interface_tokens)
    if goal and interface: return "mixed_gain"
    if goal: return "goal_gain"
    if interface: return "interface_gain"
    return "none"


def _goal_progress(current: set[str], following: set[str], goal: set[str]) -> str:
    delta = len(following & goal) - len(current & goal)
    if delta < 0: return "regress"
    if delta == 0: return "steady"
    if delta <= 2: return "advance_small"
    return "advance_large"


def _execution_status(outcome: Mapping[str, object]) -> str:
    error = bool(outcome["execution_error"])
    output = bool(outcome["output_nonempty"])
    continues = bool(outcome["trajectory_continues"])
    if error:
        if output or not continues:
            raise ValueError("unsupported execution-error combination")
        return "error_empty_continue"
    if output: return "productive_continue" if continues else "productive_stop"
    return "empty_continue" if continues else "empty_stop"


def adjacent_event_pairs(events: Sequence[Mapping[str, object]]):
    by_trajectory = defaultdict(list)
    for event in events: by_trajectory[str(event["trajectory_id"])].append(event)
    for trajectory in sorted(by_trajectory):
        rows = sorted(by_trajectory[trajectory], key=lambda row: int(row["step_id"]))
        for current, following in zip(rows, rows[1:]):
            if int(following["step_id"]) != int(current["step_id"]) + 1:
                raise ValueError("non-consecutive trajectory steps")
            if not current["observed_outcome"]["trajectory_continues"]:
                raise ValueError("adjacent event follows a stopped trajectory")
            yield current, following


def build_factorized_transition_rows(events: Sequence[Mapping[str, object]]) -> list[dict]:
    rows = []
    for current, following in adjacent_event_pairs(events):
        current_causal = current["causal_model_input"]
        following_causal = following["causal_model_input"]
        current_observation = _tokens(current_causal["visible_observation"])
        following_observation = _tokens(following_causal["visible_observation"])
        goal = _tokens(current_causal["trusted_goal"])
        interface = _schema_tokens(current_causal)
        new = following_observation - current_observation
        labels = {
            "observation_delta": _observation_delta(current_observation, following_observation),
            "evidence_delta": _evidence_delta(new, goal, interface),
            "goal_progress": _goal_progress(current_observation, following_observation, goal),
            "execution_status": _execution_status(current["observed_outcome"]),
        }
        for factor, label in labels.items():
            if label not in FACTOR_CLASSES[factor]: raise AssertionError((factor, label))
        fingerprint = hashlib.sha256(
            json.dumps(labels, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        rows.append({
            "source_event_id": current["event_id"],
            "target_event_id": following["event_id"],
            "trajectory_id": current["trajectory_id"],
            "task_name": current["task_name"],
            "step_id": current["step_id"],
            "labels": labels,
            "label_fingerprint": fingerprint,
        })
    return rows


def audit_factorized_transition_rows(rows: Sequence[Mapping[str, object]], *, expected_tasks: int) -> dict:
    counts = {factor: Counter(row["labels"][factor] for row in rows) for factor in FACTOR_CLASSES}
    tasks = sorted({str(row["task_name"]) for row in rows})
    trajectories = {str(row["trajectory_id"]) for row in rows}
    duplicate_sources = len(rows) - len({str(row["source_event_id"]) for row in rows})
    class_fractions = {
        factor: max(values.values()) / len(rows) for factor, values in counts.items()
    }
    checks = {
        "expected_adjacent_rows": len(rows) == 4703,
        "all_tasks_covered": len(tasks) == expected_tasks,
        "unique_source_events": duplicate_sources == 0,
        "all_factors_multiclass": all(len(values) >= 2 for values in counts.values()),
        "no_factor_above_98pct": all(value <= 0.98 for value in class_fractions.values()),
        "fingerprints_valid": all(
            row["label_fingerprint"] == hashlib.sha256(
                json.dumps(row["labels"], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest() for row in rows
        ),
        "labels_exclude_action_and_outcome_names": all(
            not ({"next_action", "task_success", "attack_success", "joint_outcome"} & set(row["labels"]))
            for row in rows
        ),
    }
    return {
        "passed": all(checks.values()), "checks": checks, "rows": len(rows),
        "tasks": len(tasks), "trajectories": len(trajectories),
        "class_counts": {factor: dict(values) for factor, values in counts.items()},
        "maximum_class_fraction": class_fractions,
        "duplicate_source_events": duplicate_sources,
        "model_input_fields": ["causal_model_input at source event"],
        "target_sources": ["next visible observation", "current observed execution outcome"],
        "uses_final_joint_outcome": False, "uses_next_action_as_factor_label": False,
    }
