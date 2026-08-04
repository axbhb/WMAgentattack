"""Label-blind AgentDojo state instrumentation for clean trajectories.

The adapter records exact in-memory state transitions around tool execution.  It
does not copy raw tool output, final utility/security labels, or attack metadata.
Ground-truth calls can be converted into *target-only* goal slots for progress
audits, but those slots are explicitly separate from victim-dynamics inputs.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .decision_state import canonical_json_value, stable_fingerprint
from .exact_simulator import canonical_state_delta


CLEAN_STATE_SCHEMA_VERSION = "wmagentattack.clean_state.v1"


class ArgumentEntityLink(BaseModel):
    model_config = ConfigDict(extra="forbid")

    argument_path: str
    value_fingerprint: str
    state_paths: tuple[str, ...]
    resolution: Literal["no_match", "unique", "ambiguous"]


class CleanStateTransition(BaseModel):
    """One exact tool transition without post-rollout outcome labels."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = CLEAN_STATE_SCHEMA_VERSION
    event_index: int = Field(ge=0)
    tool_schema_id: str
    normalized_argument_slots: tuple[str, ...]
    argument_entity_links: tuple[ArgumentEntityLink, ...]
    tool_execution_status: Literal["success", "error"]
    tool_error_type: str | None
    tool_output_type: str
    canonical_state_before: Any
    canonical_state_after: Any
    state_before_fingerprint: str
    state_after_fingerprint: str
    canonical_state_delta: tuple[dict[str, Any], ...]
    state_changed: bool
    task_progress_delta: float | None = None
    irreversible_effect: bool | None = None
    unavailable_fields: tuple[str, ...] = (
        "task_progress_delta",
        "irreversible_effect",
    )
    outcome_labels_present: bool = False


class GroundTruthGoalSlot(BaseModel):
    """Expert-plan target used for clean progress auditing, never model input."""

    model_config = ConfigDict(extra="forbid")

    slot_id: str
    order_index: int = Field(ge=0)
    tool_schema_id: str
    normalized_argument_slots: tuple[str, ...]
    argument_entity_links: tuple[ArgumentEntityLink, ...]
    call_signature: str
    target_only: bool = True


def _json_pointer(parts: tuple[str, ...]) -> str:
    if not parts:
        return ""
    escaped = [part.replace("~", "~0").replace("/", "~1") for part in parts]
    return "/" + "/".join(escaped)


def _canonical_state(value: Any) -> Any:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return canonical_json_value(value)


def _argument_payload(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, BaseModel):
        arguments = arguments.model_dump(mode="json")
    if not isinstance(arguments, Mapping):
        raise TypeError("tool arguments must be a mapping")
    return canonical_json_value(dict(arguments))


def _leaf_rows(value: Any, path: tuple[str, ...] = ()):  # noqa: ANN202
    if isinstance(value, Mapping):
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            yield from _leaf_rows(item, (*path, str(key)))
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            yield from _leaf_rows(item, (*path, str(index)))
        return
    yield path, value


def normalized_argument_slots(arguments: Any) -> tuple[str, ...]:
    payload = _argument_payload(arguments)
    return tuple(
        _json_pointer(path)
        for path, _ in _leaf_rows(payload)
        if path
    )


def _scalar_key(value: Any) -> str | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return json.dumps(
            {"type": type(value).__name__, "value": value},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return None


def infer_argument_entity_links(
    arguments: Any,
    canonical_state: Any,
) -> tuple[ArgumentEntityLink, ...]:
    """Link exact scalar argument values to matching canonical state paths.

    Ambiguity is retained explicitly; the adapter never guesses which repeated
    state value is the intended entity.
    """

    state_index: dict[str, list[str]] = {}
    for state_path, value in _leaf_rows(_canonical_state(canonical_state)):
        key = _scalar_key(value)
        if key is not None:
            state_index.setdefault(key, []).append(_json_pointer(state_path))

    links = []
    for argument_path, value in _leaf_rows(_argument_payload(arguments)):
        key = _scalar_key(value)
        if key is None or not argument_path:
            continue
        matches = tuple(sorted(state_index.get(key, [])))
        resolution: Literal["no_match", "unique", "ambiguous"]
        if not matches:
            resolution = "no_match"
        elif len(matches) == 1:
            resolution = "unique"
        else:
            resolution = "ambiguous"
        links.append(
            ArgumentEntityLink(
                argument_path=_json_pointer(argument_path),
                value_fingerprint=stable_fingerprint(value),
                state_paths=matches,
                resolution=resolution,
            )
        )
    return tuple(links)


def canonical_call_signature(function: str, arguments: Any) -> str:
    """Fingerprint an expert or observed call for target-only slot matching."""

    return stable_fingerprint(
        {"function": str(function), "arguments": _argument_payload(arguments)}
    )


def _call_parts(call: Any) -> tuple[str, Any]:
    if isinstance(call, Mapping):
        return str(call["function"]), call["args"]
    return str(call.function), call.args


def build_ground_truth_goal_slots(
    calls: Sequence[Any],
    initial_state: Any,
) -> tuple[GroundTruthGoalSlot, ...]:
    slots = []
    state = _canonical_state(initial_state)
    for index, call in enumerate(calls):
        function, arguments = _call_parts(call)
        slots.append(
            GroundTruthGoalSlot(
                slot_id=f"goal-slot-{index:03d}",
                order_index=index,
                tool_schema_id=function,
                normalized_argument_slots=normalized_argument_slots(arguments),
                argument_entity_links=infer_argument_entity_links(arguments, state),
                call_signature=canonical_call_signature(function, arguments),
            )
        )
    return tuple(slots)


def match_completed_goal_slots(
    observed_calls: Sequence[Any],
    goal_slots: Sequence[GroundTruthGoalSlot],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Greedily match exact calls to target-only slots without double counting."""

    unmatched = list(goal_slots)
    completed = []
    for call in observed_calls:
        function, arguments = _call_parts(call)
        signature = canonical_call_signature(function, arguments)
        for index, slot in enumerate(unmatched):
            if slot.call_signature == signature:
                completed.append(slot.slot_id)
                unmatched.pop(index)
                break
    return tuple(completed), tuple(slot.slot_id for slot in unmatched)


def instrument_function_call(
    runtime: Any,
    environment: Any,
    *,
    event_index: int,
    function: str,
    arguments: Any,
) -> tuple[CleanStateTransition, Any]:
    """Execute one exact in-memory function and capture a label-blind diff."""

    before = _canonical_state(copy.deepcopy(environment))
    links = infer_argument_entity_links(arguments, before)
    output, error = runtime.run_function(environment, function, arguments)
    after = _canonical_state(copy.deepcopy(environment))
    delta = canonical_state_delta(before, after)
    error_type = None
    if error:
        error_type = str(error).split(":", 1)[0]
    transition = CleanStateTransition(
        event_index=event_index,
        tool_schema_id=str(function),
        normalized_argument_slots=normalized_argument_slots(arguments),
        argument_entity_links=links,
        tool_execution_status="error" if error else "success",
        tool_error_type=error_type,
        tool_output_type=type(output).__name__,
        canonical_state_before=before,
        canonical_state_after=after,
        state_before_fingerprint=stable_fingerprint(before),
        state_after_fingerprint=stable_fingerprint(after),
        canonical_state_delta=delta,
        state_changed=bool(delta),
    )
    return transition, output


def candidate_tool_manifest(runtime: Any) -> tuple[dict[str, Any], ...]:
    """Return the suite-wide callable manifest; no dynamic preconditions exist."""

    rows = []
    for name, function in sorted(runtime.functions.items()):
        parameters = function.parameters.model_json_schema()
        rows.append(
            {
                "name": str(name),
                "description": str(function.description),
                "parameters": canonical_json_value(parameters),
                "dependencies": sorted(str(key) for key in function.dependencies),
                "precondition_metadata_available": False,
                "irreversibility_metadata_available": False,
            }
        )
    return tuple(rows)
