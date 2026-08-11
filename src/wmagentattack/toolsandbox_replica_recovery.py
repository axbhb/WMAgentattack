"""Label-blind canonicalization for ToolSandbox replica measurement recovery.

The original exact replicas and their raw hashes remain immutable.  This
module only removes two representation-level values that are unavailable to a
world model and have no task semantics: Python object memory addresses in
exception messages and UUIDs created during the replay itself.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from wmagentattack.multisource_semantic_data import stable_hash


MEMORY_ADDRESS_RE = re.compile(r"(?<=\bat )0x[0-9A-Fa-f]+\b")
UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}"
)


def _uuid_paths(value: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], str]]:
    found: list[tuple[tuple[str, ...], str]] = []
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            found.extend(_uuid_paths(value[key], (*path, str(key))))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_uuid_paths(child, (*path, str(index))))
    elif isinstance(value, str) and UUID_RE.fullmatch(value):
        found.append((path, value.lower()))
    return found


def runtime_uuid_mapping(before: Any, after: Any, output: Any) -> dict[str, str]:
    """Map only UUIDs introduced after execution to deterministic placeholders."""

    before_ids = {identifier for _, identifier in _uuid_paths(before)}
    candidates = [
        item
        for item in [*_uuid_paths(after), *_uuid_paths(output, ("output",))]
        if item[1] not in before_ids
    ]
    first_path: dict[str, tuple[str, ...]] = {}
    for path, identifier in candidates:
        first_path.setdefault(identifier, path)
    ordered = sorted(first_path, key=lambda identifier: (first_path[identifier], identifier))
    return {
        identifier: f"<RUNTIME_UUID_{index}>"
        for index, identifier in enumerate(ordered)
    }


def _replace(value: Any, uuid_mapping: Mapping[str, str]) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _replace(child, uuid_mapping) for key, child in value.items()}
    if isinstance(value, list):
        return [_replace(child, uuid_mapping) for child in value]
    if isinstance(value, tuple):
        return [_replace(child, uuid_mapping) for child in value]
    if isinstance(value, str):
        exact_uuid = UUID_RE.fullmatch(value)
        if exact_uuid is not None:
            replacement = uuid_mapping.get(value.lower())
            if replacement is not None:
                return replacement
        return MEMORY_ADDRESS_RE.sub("<MEMORY_ADDRESS>", value)
    return value


def canonical_replica_payload(
    *,
    before: Any,
    after: Any,
    output: Any,
    error: Any,
    status: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Return one comparison payload plus an auditable normalization count."""

    uuid_mapping = runtime_uuid_mapping(before, after, output)
    raw_text = str(error) if error is not None else ""
    memory_addresses = len(MEMORY_ADDRESS_RE.findall(raw_text))
    canonical_before = _replace(before, uuid_mapping)
    canonical_after = _replace(after, uuid_mapping)
    payload = {
        "status": status,
        "error": _replace(error, uuid_mapping),
        "output": _replace(output, uuid_mapping),
        "state_before_sha256": stable_hash(canonical_before),
        "state_after_sha256": stable_hash(canonical_after),
        "state_changed": stable_hash(before) != stable_hash(after),
    }
    return payload, {
        "memory_addresses": memory_addresses,
        "runtime_uuids": len(uuid_mapping),
    }


def replicas_semantically_identical(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return stable_hash(first) == stable_hash(second)
