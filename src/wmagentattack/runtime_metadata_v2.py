"""Episode-local normalization of explicitly registered simulator metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict

from .decision_state import canonical_json_value, stable_fingerprint


class RuntimeMetadataRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    volatile_fields: tuple[str, ...]
    origin: str
    normalization: str
    raw_exact_state_retained: bool


class RuntimeMetadataRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    status: str
    benchmark_version: str
    suite: str
    rules: dict[str, RuntimeMetadataRule]
    source_evidence: tuple[dict[str, str], ...]
    hard_boundaries: dict[str, bool]


def load_runtime_metadata_registry(path: Path) -> RuntimeMetadataRegistry:
    return RuntimeMetadataRegistry.model_validate_json(path.read_text(encoding="utf-8"))


def _encoded(value: Any) -> str:
    return json.dumps(
        canonical_json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _discover_field_values(value: Any, fields: frozenset[str]) -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []

    def visit(item: Any) -> None:
        canonical = canonical_json_value(item)
        if isinstance(canonical, Mapping):
            path = str(canonical.get("path", ""))
            path_field = path.rstrip("/").rsplit("/", 1)[-1] if path else ""
            if path_field in fields and "value" in canonical:
                rows.append((path_field, canonical["value"]))
            attribute_name = str(canonical.get("name", ""))
            if attribute_name in fields and "value" in canonical:
                rows.append((attribute_name, canonical["value"]))
            for key, child in canonical.items():
                if str(key) in fields:
                    rows.append((str(key), child))
                visit(child)
        elif isinstance(canonical, list):
            for child in canonical:
                visit(child)

    visit(value)
    return rows


class EpisodeLocalMetadataNormalizer:
    """Replace only observed registered values in semantic replay projections."""

    def __init__(self, registry: RuntimeMetadataRegistry) -> None:
        self.registry = registry
        self._bindings: dict[str, dict[str, str]] = {}

    @property
    def binding_count(self) -> int:
        return sum(len(values) for values in self._bindings.values())

    def observe_transition(
        self,
        *,
        tool_name: str,
        call_index: int,
        runtime_output: Any,
        exact_delta: Any,
    ) -> tuple[str, ...]:
        rule = self.registry.rules.get(tool_name)
        if rule is None:
            return ()
        fields = frozenset(rule.volatile_fields)
        discovered = [
            *_discover_field_values(runtime_output, fields),
            *_discover_field_values(exact_delta, fields),
        ]
        newly_bound = []
        per_field_ordinal: dict[str, int] = {}
        for field, raw_value in discovered:
            encoded = _encoded(raw_value)
            field_bindings = self._bindings.setdefault(field, {})
            if encoded in field_bindings:
                continue
            ordinal = per_field_ordinal.get(field, 0)
            per_field_ordinal[field] = ordinal + 1
            token = f"VOLATILE::{field}::CALL_{call_index:03d}::ITEM_{ordinal:03d}"
            field_bindings[encoded] = token
            newly_bound.append(token)
        return tuple(newly_bound)

    def _replace(self, field: str, value: Any) -> Any:
        token = self._bindings.get(field, {}).get(_encoded(value))
        return token if token is not None else canonical_json_value(value)

    def normalize(self, value: Any) -> Any:
        canonical = canonical_json_value(value)
        if isinstance(canonical, list):
            return [self.normalize(item) for item in canonical]
        if not isinstance(canonical, Mapping):
            return canonical
        output = {}
        path = str(canonical.get("path", ""))
        path_field = path.rstrip("/").rsplit("/", 1)[-1] if path else ""
        attribute_name = str(canonical.get("name", ""))
        for key, item in canonical.items():
            field = str(key)
            if field in self._bindings:
                output[field] = self._replace(field, item)
            elif field == "value" and path_field in self._bindings:
                output[field] = self._replace(path_field, item)
            elif field == "value" and attribute_name in self._bindings:
                output[field] = self._replace(attribute_name, item)
            else:
                output[field] = self.normalize(item)
        return output

    def semantic_fingerprint(self, value: Any) -> str:
        return stable_fingerprint(self.normalize(value))

    def public_manifest(self) -> dict[str, Any]:
        return {
            "binding_count": self.binding_count,
            "fields": {
                field: tuple(sorted(bindings.values()))
                for field, bindings in sorted(self._bindings.items())
            },
            "raw_values_retained_only_in_exact_artifacts": True,
        }
