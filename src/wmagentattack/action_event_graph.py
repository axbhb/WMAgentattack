"""Causal, value-anonymized action-event graphs for AgentDojo trajectories."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence


_TOKEN = re.compile(r"[a-z][a-z0-9_]+", re.I)
_EMAIL = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.I)
_DATE = re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b")
_NUMBER = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?")
_YAML_KEY = re.compile(r"(?m)^\s*(?:-\s*)?([A-Za-z_][\w -]{0,48}):")
_FUNCTION_TAG = re.compile(r"<function[=>]([A-Za-z_][\w]*)", re.I)
_STOP = frozenset(
    "a an and are as at be been by can could did do does for from get given has have if in "
    "into is it its may of on or return that the their them these this to type use used user "
    "using was were when where which will with would information data value values".split()
)

_KEY_GROUPS = (
    ("identifier", re.compile(r"(^|_)(id|uuid|identifier|number)($|_)", re.I)),
    ("person", re.compile(r"name|participant|recipient|sender|author|contact|user", re.I)),
    ("location", re.compile(r"address|street|city|location|country|destination|origin", re.I)),
    ("time", re.compile(r"date|day|time|year|month|start|end|duration", re.I)),
    ("communication", re.compile(r"email|phone|subject|body|message|content|description", re.I)),
    ("money", re.compile(r"amount|price|budget|balance|currency|account", re.I)),
    ("status", re.compile(r"status|state|error|success|confirmed|open", re.I)),
    ("rating", re.compile(r"rating|review|score", re.I)),
    ("resource", re.compile(r"file|url|channel|hotel|restaurant|flight|event|transaction", re.I)),
    ("kind", re.compile(r"type|kind|category|format", re.I)),
)


def _slug(value: object) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(value).lower()).strip("_") or "none"


def _key_group(value: object) -> str:
    key = _slug(value)
    for name, pattern in _KEY_GROUPS:
        if pattern.search(key):
            return name
    return "other"


def _value_type(value: object) -> str:
    if value is None: return "null"
    if isinstance(value, bool): return "boolean"
    if isinstance(value, (int, float)): return "number"
    if isinstance(value, Mapping): return "mapping"
    if isinstance(value, (list, tuple, set)): return "sequence"
    text = str(value)
    if _EMAIL.fullmatch(text): return "email"
    if _DATE.fullmatch(text): return "date"
    if _URL.match(text): return "url"
    return "string"


def _bin(value: int) -> str:
    if value == 0: return "0"
    if value == 1: return "1"
    if value <= 3: return "2_3"
    if value <= 7: return "4_7"
    return "8_plus"


def _tokens(value: object) -> set[str]:
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return {
        piece for token in _TOKEN.findall(value.lower()) for piece in token.split("_")
        if len(piece) >= 3 and piece not in _STOP
    }


def _overlap_bin(left: set[str], right: set[str]) -> str:
    count = len(left & right)
    if count == 0: return "none"
    if count <= 2: return "small"
    if count <= 5: return "medium"
    return "large"


def _parse_receipt(text: str):
    stripped = text.strip()
    if not stripped:
        return "empty", None
    if stripped.startswith("ERROR:"):
        return "error_text", None
    try:
        value = json.loads(stripped)
    except Exception:
        try:
            value = ast.literal_eval(stripped)
        except Exception:
            value = None
    if isinstance(value, Mapping): return "mapping", value
    if isinstance(value, (list, tuple)): return "sequence", value
    if value is not None: return "scalar", value
    if _FUNCTION_TAG.search(stripped): return "function_tag_text", None
    if _YAML_KEY.search(stripped): return "yaml_like", None
    return "free_text", None


def _walk(value: object, *, depth: int = 0):
    key_groups = Counter(); value_types = Counter(); nodes = 1; maximum_depth = depth
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_groups[_key_group(key)] += 1
            child_keys, child_types, child_nodes, child_depth = _walk(child, depth=depth + 1)
            key_groups.update(child_keys); value_types.update(child_types)
            nodes += child_nodes; maximum_depth = max(maximum_depth, child_depth)
    elif isinstance(value, (list, tuple)):
        for child in value:
            child_keys, child_types, child_nodes, child_depth = _walk(child, depth=depth + 1)
            key_groups.update(child_keys); value_types.update(child_types)
            nodes += child_nodes; maximum_depth = max(maximum_depth, child_depth)
    else:
        value_types[_value_type(value)] += 1
    return key_groups, value_types, nodes, maximum_depth


def _receipt_summary(step: Mapping[str, object]) -> dict:
    text = str(step.get("skill_output") or "")
    receipt_format, parsed = _parse_receipt(text)
    if parsed is not None:
        key_groups, value_types, nodes, depth = _walk(parsed)
    else:
        key_groups = Counter(_key_group(key) for key in _YAML_KEY.findall(text))
        value_types = Counter()
        nodes = max(1, sum(key_groups.values()))
        depth = int(bool(key_groups))
    indicators = {
        "email": len(_EMAIL.findall(text)), "url": len(_URL.findall(text)),
        "date": len(_DATE.findall(text)), "number": len(_NUMBER.findall(text)),
        "function_tag": len(_FUNCTION_TAG.findall(text)),
    }
    return {
        "format": receipt_format,
        "key_groups": key_groups,
        "value_types": value_types,
        "nodes": nodes,
        "depth": depth,
        "indicators": indicators,
        "tokens": _tokens(text),
    }


def event_graph_features(
    step: Mapping[str, object], *, previous_step: Mapping[str, object] | None,
    current_legal: Sequence[str], previous_legal: Sequence[str] | None,
) -> tuple[list[str], dict]:
    features = set()
    skill = _slug(step.get("selected_skill") or "none")
    tool = _slug(step.get("selected_tool") or "none")
    features.update((f"action.skill={skill}", f"action.tool={tool}"))
    arguments = step.get("skill_arguments") or {}
    if not isinstance(arguments, Mapping):
        raise ValueError("skill_arguments must be a mapping")
    features.add(f"action.argument_count={_bin(len(arguments))}")
    argument_tokens = set()
    argument_groups = Counter()
    for key, value in arguments.items():
        group = _key_group(key); argument_groups[group] += 1
        features.add(f"action.argument_group={group}")
        features.add(f"action.argument_type={_value_type(value)}")
        argument_tokens |= _tokens(value)
    receipt = _receipt_summary(step)
    features.add(f"receipt.format={receipt['format']}")
    features.add(f"receipt.node_count={_bin(receipt['nodes'])}")
    features.add(f"receipt.depth={_bin(receipt['depth'])}")
    features.add(f"receipt.error={bool(step.get('tool_error')) or receipt['format']=='error_text'}")
    for group in receipt["key_groups"]:
        features.add(f"receipt.key_group={group}")
    for value_type in receipt["value_types"]:
        features.add(f"receipt.value_type={value_type}")
    for name, count in receipt["indicators"].items():
        features.add(f"receipt.{name}_count={_bin(count)}")
    goal_tokens = _tokens(step.get("user_goal") or "")
    features.add(f"relation.goal_argument_overlap={_overlap_bin(goal_tokens, argument_tokens)}")
    features.add(f"relation.goal_receipt_overlap={_overlap_bin(goal_tokens, receipt['tokens'])}")
    features.add(
        f"relation.argument_receipt_schema_overlap={_bin(len(set(argument_groups) & set(receipt['key_groups'])))}"
    )
    history = step.get("previous_skills") or []
    features.add(f"history.length={_bin(len(history))}")
    if history:
        features.add(f"history.last_skill={_slug(history[-1])}")
    current_legal_set = set(current_legal)
    features.add(f"legal.current_count={_bin(len(current_legal_set))}")
    if previous_legal is None:
        features.add("legal.change_from_previous=initial")
    else:
        previous_legal_set = set(previous_legal)
        features.add(f"legal.enabled_from_previous={_bin(len(current_legal_set-previous_legal_set))}")
        features.add(f"legal.disabled_from_previous={_bin(len(previous_legal_set-current_legal_set))}")
    previous_receipt = _receipt_summary(previous_step) if previous_step is not None else None
    if previous_receipt is None:
        features.add("entity_schema_delta=initial")
        added = removed = set()
    else:
        current_groups = set(receipt["key_groups"]); previous_groups = set(previous_receipt["key_groups"])
        added = current_groups - previous_groups; removed = previous_groups - current_groups
        features.add(f"entity_schema_added={_bin(len(added))}")
        features.add(f"entity_schema_removed={_bin(len(removed))}")
        for group in added: features.add(f"entity_schema_added_group={group}")
        for group in removed: features.add(f"entity_schema_removed_group={group}")
    summary = {
        "selected_tool_present": bool(step.get("selected_tool")),
        "argument_count": len(arguments), "receipt_format": receipt["format"],
        "receipt_key_groups": sorted(receipt["key_groups"]),
        "entity_schema_added": sorted(added), "entity_schema_removed": sorted(removed),
    }
    return sorted(features), summary


def build_action_event_graph_dataset(steps: Sequence[Mapping[str, object]], events):
    event_by_key = {(row["trajectory_id"], int(row["step_id"])): row for row in events}
    if len(event_by_key) != len(events): raise ValueError("duplicate event keys")
    step_by_trajectory = defaultdict(list)
    for step in steps: step_by_trajectory[str(step["trajectory_id"])].append(step)
    rows = []
    for trajectory_id in sorted(step_by_trajectory):
        trajectory = sorted(step_by_trajectory[trajectory_id], key=lambda row: int(row["step_id"]))
        previous_step = None; previous_legal = None
        for step in trajectory:
            key = (trajectory_id, int(step["step_id"]))
            event = event_by_key.get(key)
            if event is None: raise ValueError(f"missing canonical event {key}")
            features, summary = event_graph_features(
                step, previous_step=previous_step,
                current_legal=event["current_legal_candidate_ids"], previous_legal=previous_legal,
            )
            fingerprint = hashlib.sha256(
                json.dumps(features, separators=(",", ":")).encode()
            ).hexdigest()
            rows.append({
                "event_id": event["event_id"], "trajectory_id": trajectory_id,
                "step_id": int(step["step_id"]), "task_name": event["task_name"],
                "features": features, "summary": summary, "feature_fingerprint": fingerprint,
            })
            previous_step = step; previous_legal = event["current_legal_candidate_ids"]
    if len(rows) != len(events): raise ValueError("step/event row mismatch")
    catalog = sorted({feature for row in rows for feature in row["features"]})
    return {
        "schema_version": "wmagentattack.action_event_graph.v1",
        "rows": rows, "feature_catalog": catalog,
    }


def audit_action_event_graph_dataset(dataset, *, expected_rows: int, expected_tasks: int):
    rows = dataset["rows"]; catalog = dataset["feature_catalog"]
    forbidden = ("task_success", "attack_success", "target_skill", "next_action", "trajectory_id=")
    counts = Counter(row["summary"]["receipt_format"] for row in rows)
    fingerprints = Counter(row["feature_fingerprint"] for row in rows)
    tool_rows = [row for row in rows if any(f.startswith("action.skill=") and f != "action.skill=finish" for f in row["features"])]
    checks = {
        "expected_rows": len(rows) == expected_rows,
        "all_tasks": len({row["task_name"] for row in rows}) == expected_tasks,
        "unique_events": len({row["event_id"] for row in rows}) == len(rows),
        "all_graphs_nonempty": all(row["features"] for row in rows),
        "catalog_exact": catalog == sorted({feature for row in rows for feature in row["features"]}),
        "fingerprints_valid": all(row["feature_fingerprint"] == hashlib.sha256(json.dumps(row["features"],separators=(",", ":")).encode()).hexdigest() for row in rows),
        "forbidden_features_absent": not any(token in feature for feature in catalog for token in forbidden),
        "tool_identity_complete_nonfinish": all(row["summary"]["selected_tool_present"] for row in tool_rows),
        "receipt_format_support": len(counts) >= 5,
        "signature_not_collapsed": max(fingerprints.values()) / len(rows) <= 0.20,
    }
    return {
        "passed": all(checks.values()), "checks": checks, "rows": len(rows),
        "tasks": len({row["task_name"] for row in rows}),
        "trajectories": len({row["trajectory_id"] for row in rows}),
        "feature_catalog_size": len(catalog), "receipt_formats": dict(counts),
        "maximum_signature_fraction": max(fingerprints.values()) / len(rows),
        "forbidden_source_fields": ["task_success", "attack_success", "target_skill", "target_skill_success", "policy_violation", "final probability labels"],
        "causal_source_fields": ["selected_skill", "selected_tool", "skill_arguments", "skill_output", "tool_error", "previous_skills", "user_goal", "current legal actions"],
    }
