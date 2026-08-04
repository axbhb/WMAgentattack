"""Content-addressed exact state storage and hard model-tower visibility rules."""

from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .decision_state import canonical_json_value, stable_fingerprint


STATE_STORAGE_SCHEMA_VERSION = "wmagentattack.state_storage.v2"


class VisibilityScope(str, Enum):
    VICTIM_OBSERVED = "VICTIM_OBSERVED"
    PLANNER_PRIVILEGED = "PLANNER_PRIVILEGED"
    SIMULATOR_INTERNAL = "SIMULATOR_INTERNAL"


class ModelTower(str, Enum):
    VICTIM_PROPOSAL = "VICTIM_PROPOSAL"
    KNOWLEDGE_PROGRESS = "KNOWLEDGE_PROGRESS"
    ENVIRONMENT_PROGRESS = "ENVIRONMENT_PROGRESS"
    COMPLETION_VALUE = "COMPLETION_VALUE"
    PLANNER_VALUE = "PLANNER_VALUE"
    SIMULATOR = "SIMULATOR"


TOWER_ACCESS: dict[ModelTower, frozenset[VisibilityScope]] = {
    ModelTower.VICTIM_PROPOSAL: frozenset({VisibilityScope.VICTIM_OBSERVED}),
    ModelTower.KNOWLEDGE_PROGRESS: frozenset({VisibilityScope.VICTIM_OBSERVED}),
    ModelTower.ENVIRONMENT_PROGRESS: frozenset(
        {VisibilityScope.VICTIM_OBSERVED, VisibilityScope.PLANNER_PRIVILEGED}
    ),
    ModelTower.COMPLETION_VALUE: frozenset(
        {VisibilityScope.VICTIM_OBSERVED, VisibilityScope.PLANNER_PRIVILEGED}
    ),
    ModelTower.PLANNER_VALUE: frozenset(
        {VisibilityScope.VICTIM_OBSERVED, VisibilityScope.PLANNER_PRIVILEGED}
    ),
    ModelTower.SIMULATOR: frozenset(VisibilityScope),
}


class StateBlobReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = STATE_STORAGE_SCHEMA_VERSION
    fingerprint: str = Field(min_length=64, max_length=64)
    storage_key: str
    byte_length: int = Field(ge=0)
    visibility_scope: Literal[VisibilityScope.SIMULATOR_INTERNAL] = (
        VisibilityScope.SIMULATOR_INTERNAL
    )


class StateBlob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = STATE_STORAGE_SCHEMA_VERSION
    fingerprint: str = Field(min_length=64, max_length=64)
    canonical_state: Any


class ExactStateTransitionV2(BaseModel):
    """Label-blind references around one exactly executed simulator call."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = STATE_STORAGE_SCHEMA_VERSION
    episode_id: str
    call_index: int = Field(ge=0)
    initial_state_ref: StateBlobReference
    state_before_ref: StateBlobReference
    state_after_ref: StateBlobReference
    exact_delta: tuple[dict[str, Any], ...]
    delta_operation_count: int = Field(ge=0)
    delta_roots: tuple[str, ...]
    execution_status: Literal["success", "error"]
    error_type: str | None = None
    state_changed: bool
    outcome_labels_present: bool = False


class ScopedFeature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    visibility_scope: VisibilityScope
    value: Any
    outcome_labels_present: bool = False


class FeatureEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = STATE_STORAGE_SCHEMA_VERSION
    fields: tuple[ScopedFeature, ...]
    outcome_labels_present: bool = False

    def view(
        self,
        tower: ModelTower,
        *,
        requested_fields: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Return a tower view and fail closed on any forbidden request."""

        by_name = {field.name: field for field in self.fields}
        if len(by_name) != len(self.fields):
            raise ValueError("feature envelope contains duplicate names")
        names = tuple(requested_fields) if requested_fields is not None else tuple(by_name)
        missing = sorted(set(names) - set(by_name))
        if missing:
            raise KeyError(f"unknown feature fields: {missing}")
        forbidden = [
            name
            for name in names
            if by_name[name].visibility_scope not in TOWER_ACCESS[tower]
        ]
        if forbidden:
            scopes = {name: by_name[name].visibility_scope.value for name in forbidden}
            raise PermissionError(
                f"tower {tower.value} cannot access requested fields: {scopes}"
            )
        return {name: by_name[name].value for name in names}

    def allowed_view(self, tower: ModelTower) -> dict[str, Any]:
        """Return only fields explicitly allowed for a tower."""

        return {
            field.name: field.value
            for field in self.fields
            if field.visibility_scope in TOWER_ACCESS[tower]
        }


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        canonical_json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class ContentAddressedStateStore:
    """Filesystem state store that writes each exact canonical state once."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.blob_root = self.root / "blobs"
        self.blob_root.mkdir(parents=True, exist_ok=True)

    def _path(self, fingerprint: str) -> Path:
        return self.blob_root / fingerprint[:2] / f"{fingerprint}.json"

    def put(self, state: Any) -> StateBlobReference:
        canonical = canonical_json_value(state)
        fingerprint = stable_fingerprint(canonical)
        blob = StateBlob(fingerprint=fingerprint, canonical_state=canonical)
        encoded = _canonical_bytes(blob.model_dump(mode="json"))
        path = self._path(fingerprint)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != encoded:
                raise ValueError(f"content-address collision at {fingerprint}")
        else:
            temporary = path.with_suffix(f".tmp-{os.getpid()}")
            temporary.write_bytes(encoded)
            os.replace(temporary, path)
        return StateBlobReference(
            fingerprint=fingerprint,
            storage_key=path.relative_to(self.root).as_posix(),
            byte_length=len(encoded),
        )

    def get(
        self, reference: StateBlobReference, *, requesting_tower: ModelTower
    ) -> Any:
        if VisibilityScope.SIMULATOR_INTERNAL not in TOWER_ACCESS[requesting_tower]:
            raise PermissionError(
                f"tower {requesting_tower.value} cannot dereference internal state blobs"
            )
        path = self.root / reference.storage_key
        encoded = path.read_bytes()
        if len(encoded) != reference.byte_length:
            raise ValueError("state blob byte-length mismatch")
        blob = StateBlob.model_validate_json(encoded)
        if blob.fingerprint != reference.fingerprint:
            raise ValueError("state blob reference mismatch")
        if stable_fingerprint(blob.canonical_state) != reference.fingerprint:
            raise ValueError("state blob content fingerprint mismatch")
        return blob.canonical_state


def _delta_roots(delta: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    roots = set()
    for operation in delta:
        path = str(operation.get("path", ""))
        roots.add("/" + path.lstrip("/").split("/", 1)[0] if path else "<root>")
    return tuple(sorted(roots))


def build_exact_state_transition(
    store: ContentAddressedStateStore,
    *,
    episode_id: str,
    call_index: int,
    initial_state: Any,
    state_before: Any,
    state_after: Any,
    exact_delta: Sequence[Mapping[str, Any]],
    execution_status: Literal["success", "error"],
    error_type: str | None,
) -> ExactStateTransitionV2:
    canonical_delta = tuple(canonical_json_value(dict(row)) for row in exact_delta)
    return ExactStateTransitionV2(
        episode_id=episode_id,
        call_index=call_index,
        initial_state_ref=store.put(initial_state),
        state_before_ref=store.put(state_before),
        state_after_ref=store.put(state_after),
        exact_delta=canonical_delta,
        delta_operation_count=len(canonical_delta),
        delta_roots=_delta_roots(canonical_delta),
        execution_status=execution_status,
        error_type=error_type,
        state_changed=bool(canonical_delta),
    )
