"""Exact state-transition adapter for an AgentDojo-style sandbox.

The world model predicts only victim actions.  Known tool semantics, state
mutation, and benchmark checkers stay outside the learned model and are
invoked through this deterministic adapter.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .decision_state import VictimActionEvent, canonical_json_value


@dataclass(frozen=True)
class ToolExecution:
    next_state: Any
    output: Any = None


@dataclass(frozen=True)
class ExactTransition:
    state_before: Any
    state_after: Any
    state_delta: tuple[dict[str, Any], ...]
    tool_output: Any
    tool_error: str | None
    utility: bool | None
    security: bool | None


ToolExecutor = Callable[[Any, VictimActionEvent], ToolExecution]
StateCanonicalizer = Callable[[Any], Any]
StateChecker = Callable[[Any], bool]


def _json_pointer(parts: tuple[str, ...]) -> str:
    if not parts:
        return ""
    escaped = [part.replace("~", "~0").replace("/", "~1") for part in parts]
    return "/" + "/".join(escaped)


def canonical_state_delta(before: Any, after: Any) -> tuple[dict[str, Any], ...]:
    """Return a deterministic RFC-6901-style structural state diff."""

    changes: list[dict[str, Any]] = []

    def visit(left: Any, right: Any, path: tuple[str, ...]) -> None:
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            left_keys = {str(key) for key in left}
            right_keys = {str(key) for key in right}
            left_map = {str(key): value for key, value in left.items()}
            right_map = {str(key): value for key, value in right.items()}
            for key in sorted(left_keys - right_keys):
                changes.append(
                    {"op": "remove", "path": _json_pointer((*path, key)), "old": left_map[key]}
                )
            for key in sorted(right_keys - left_keys):
                changes.append(
                    {"op": "add", "path": _json_pointer((*path, key)), "value": right_map[key]}
                )
            for key in sorted(left_keys & right_keys):
                visit(left_map[key], right_map[key], (*path, key))
            return
        if (
            isinstance(left, Sequence)
            and isinstance(right, Sequence)
            and not isinstance(left, (str, bytes, bytearray))
            and not isinstance(right, (str, bytes, bytearray))
        ):
            if list(left) != list(right):
                changes.append(
                    {
                        "op": "replace",
                        "path": _json_pointer(path),
                        "old": left,
                        "value": right,
                    }
                )
            return
        if left != right:
            changes.append(
                {
                    "op": "replace",
                    "path": _json_pointer(path),
                    "old": left,
                    "value": right,
                }
            )

    visit(canonical_json_value(before), canonical_json_value(after), ())
    return tuple(changes)


class ExactSandboxSimulator:
    """Apply an action using the benchmark's exact in-memory tool executor."""

    def __init__(
        self,
        executor: ToolExecutor,
        *,
        canonicalizer: StateCanonicalizer = canonical_json_value,
        utility_checker: StateChecker | None = None,
        security_checker: StateChecker | None = None,
    ) -> None:
        self._executor = executor
        self._canonicalizer = canonicalizer
        self._utility_checker = utility_checker
        self._security_checker = security_checker

    def transition(self, state: Any, action: VictimActionEvent) -> ExactTransition:
        working_state = copy.deepcopy(state)
        before = canonical_json_value(self._canonicalizer(copy.deepcopy(working_state)))
        output: Any = None
        error: str | None = None
        next_state = working_state
        if not action.stop:
            try:
                execution = self._executor(working_state, action)
                if not isinstance(execution, ToolExecution):
                    raise TypeError("tool executor must return ToolExecution")
                next_state = execution.next_state
                output = execution.output
            except Exception as exc:  # benchmark errors are part of the transition
                error = f"{type(exc).__name__}: {exc}"
        after = canonical_json_value(self._canonicalizer(copy.deepcopy(next_state)))
        utility = self._utility_checker(next_state) if self._utility_checker else None
        security = self._security_checker(next_state) if self._security_checker else None
        return ExactTransition(
            state_before=before,
            state_after=after,
            state_delta=canonical_state_delta(before, after),
            tool_output=canonical_json_value(output),
            tool_error=error,
            utility=utility,
            security=security,
        )

