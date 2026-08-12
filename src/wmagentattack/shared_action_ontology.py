"""Label-blind shared action ontology for heterogeneous tool benchmarks.

The ontology is derived only from a candidate's public name, description,
parameter schema, and tool/text kind.  It never inspects selected actions,
outcomes, attack annotations, or held-out labels.
"""

from __future__ import annotations

import copy
import re
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from .clean_evidence_probe import hashed_text
from .multisource_suitability import stable_hash


ACTION_ONTOLOGY_VERSION = "wmagentattack.shared_action_ontology.v1"
ONTOLOGY_VECTOR_MODES = ("ontology_only", "ontology_local_residual")

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_OPERATIONS = (
    ("delete", {"cancel", "delete", "remove", "revoke"}),
    ("communicate", {"email", "message", "notify", "post", "send", "share"}),
    ("create", {"add", "book", "buy", "create", "make", "order", "reserve", "schedule", "set"}),
    ("update", {"change", "edit", "modify", "move", "rename", "update"}),
    ("compute", {"calculate", "convert", "compute", "timestamp"}),
    ("authorize", {"authenticate", "login", "verify"}),
    ("retrieve", {"check", "fetch", "find", "get", "list", "lookup", "query", "read", "retrieve", "search", "show", "view"}),
)
_OBJECTS = {
    "accommodation": {"hotel", "hotels", "accommodation"},
    "calendar": {"calendar", "event", "events", "meeting", "meetings"},
    "communication": {"email", "gmail", "mail", "message", "messages", "slack"},
    "contact": {"contact", "contacts", "phone"},
    "finance": {"account", "bank", "banking", "balance", "binance", "crypto", "payment", "transaction", "transactions"},
    "food": {"cuisine", "restaurant", "restaurants"},
    "knowledge": {"article", "document", "documents", "file", "files", "information", "record", "records"},
    "product": {"amazon", "item", "items", "order", "orders", "product", "products", "shopping"},
    "profile": {"address", "addresses", "passport", "profile", "user", "users"},
    "reminder": {"reminder", "reminders"},
    "social": {"tweet", "tweets", "twitter"},
    "time": {"date", "datetime", "day", "time", "timestamp"},
    "transport": {"car", "flight", "flights", "rental", "travel"},
    "weather": {"forecast", "weather"},
}


def _tokens(value: str) -> set[str]:
    expanded = value.replace("_", " ").replace("-", " ")
    return {token.lower() for token in _TOKEN_RE.findall(expanded)}


def _descriptor_text(descriptor: Mapping[str, Any]) -> str:
    function = descriptor["function"]
    parameters = function.get("parameters", {})
    properties = parameters.get("properties", {}) if isinstance(parameters, Mapping) else {}
    parts = [str(function.get("name", "")), str(function.get("description", ""))]
    for name, specification in sorted(properties.items()):
        parts.extend([str(name), str(specification.get("description", ""))])
    return " ".join(parts)


def action_ontology(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """Map one public candidate schema into a shared, causal ontology."""

    function = descriptor["function"]
    name = str(function.get("name", ""))
    kind = str(descriptor.get("kind", "tool"))
    tokens = _tokens(_descriptor_text(descriptor))
    terminal = kind == "text_or_stop" or name.upper() == "TEXT" or bool(
        tokens & {"finish", "stop", "conversation"}
    )
    if terminal:
        operation = "stop"
    else:
        operation = "invoke"
        name_tokens = _tokens(name)
        for candidate, keywords in _OPERATIONS:
            if name_tokens & keywords:
                operation = candidate
                break
        if operation == "invoke":
            for candidate, keywords in _OPERATIONS:
                if tokens & keywords:
                    operation = candidate
                    break

    objects = sorted(
        object_name
        for object_name, keywords in _OBJECTS.items()
        if tokens & keywords
    )
    if not objects:
        objects = ["generic"]
    if operation == "stop":
        effect = "terminal"
    elif operation in {"retrieve", "compute", "authorize"}:
        effect = "read_only"
    elif operation in {"create", "update", "delete", "communicate"}:
        effect = "mutating"
    else:
        effect = "unknown"
    if operation == "communicate" or set(objects) & {"communication", "social"}:
        communication_scope = "external"
    elif effect == "mutating":
        communication_scope = "local"
    else:
        communication_scope = "none"

    parameters = function.get("parameters", {})
    if not isinstance(parameters, Mapping):
        parameters = {}
    properties = parameters.get("properties", {})
    if not isinstance(properties, Mapping):
        properties = {}
    required = parameters.get("required", [])
    if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
        required = []
    parameter_types = sorted(
        Counter(
            str(specification.get("type", "unknown"))
            for specification in properties.values()
            if isinstance(specification, Mapping)
        ).items()
    )
    bridge = {
        "operation": operation,
        "objects": objects,
        "effect": effect,
        "communication_scope": communication_scope,
        "terminal": terminal,
    }
    return {
        "ontology_version": ACTION_ONTOLOGY_VERSION,
        **bridge,
        "argument_shape": {
            "property_count": len(properties),
            "required_count": len(required),
            "parameter_types": parameter_types,
        },
        "bridge_key": stable_hash(bridge)[:16],
    }


def ontology_candidate_vector(
    descriptor: Mapping[str, Any], *, mode: str, hash_dimension: int
) -> np.ndarray:
    if mode not in ONTOLOGY_VECTOR_MODES:
        raise ValueError(f"unsupported ontology vector mode: {mode}")
    ontology = descriptor.get("shared_action_ontology") or action_ontology(descriptor)
    shared_fields = {
        key: ontology[key]
        for key in (
            "operation",
            "objects",
            "effect",
            "communication_scope",
            "terminal",
        )
    }
    shared = hashed_text(
        shared_fields, hash_dimension, "shared-action-ontology-v1"
    )
    if mode == "ontology_only":
        return shared
    function = descriptor["function"]
    local_schema = {
        "name": function.get("name", ""),
        "description": function.get("description", ""),
        "parameters": function.get("parameters", {}),
        "kind": descriptor.get("kind", "tool"),
    }
    local = hashed_text(local_schema, hash_dimension, "source-local-schema-v1")
    vector = 0.8 * shared + 0.2 * local
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0.0 else vector


def _canonical_prior_action(
    prior: str, descriptors: Sequence[Mapping[str, Any]]
) -> str:
    for descriptor in descriptors:
        if str(descriptor["function"].get("name")) == prior:
            return str(action_ontology(descriptor)["bridge_key"])
    synthetic = {
        "kind": "text_or_stop" if prior in {"finish", "TEXT"} else "tool",
        "function": {
            "name": prior,
            "description": "",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    return str(action_ontology(synthetic)["bridge_key"])


def ontology_aligned_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return a row whose state-side action interface uses ontology keys."""

    output = copy.deepcopy(dict(row))
    causal = copy.deepcopy(dict(output["causal_model_input"]))
    descriptors = [
        {
            "kind": "tool",
            "function": schema["function"],
        }
        for schema in causal["tool_schemas"]
    ]
    causal["legal_tool_names"] = [
        str(action_ontology(descriptor)["bridge_key"])
        for descriptor in descriptors
    ]
    causal["visible_prior_tool"] = _canonical_prior_action(
        str(causal["visible_prior_tool"]), descriptors
    )
    output["causal_model_input"] = causal
    output["ontology_causal_input_fingerprint"] = stable_hash(causal)
    return output


def annotate_action_ontology(
    dataset: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = copy.deepcopy(dict(dataset))
    catalog = output["candidate_catalog"]
    for descriptor in catalog.values():
        descriptor["shared_action_ontology"] = action_ontology(descriptor)
    rows = [ontology_aligned_row(row) for row in output["rows"]]
    output["rows"] = rows
    output["shared_action_ontology_version"] = ACTION_ONTOLOGY_VERSION

    source_bridges: dict[str, set[str]] = defaultdict(set)
    source_operations: dict[str, Counter[str]] = defaultdict(Counter)
    for descriptor in catalog.values():
        source = str(descriptor["source"])
        ontology = descriptor["shared_action_ontology"]
        source_bridges[source].add(str(ontology["bridge_key"]))
        source_operations[source][str(ontology["operation"])] += 1
    agentdojo = source_bridges["agentdojo"]
    tool_sandbox = source_bridges["tool_sandbox"]
    injecagent = source_bridges["injecagent"]
    all_three = agentdojo & tool_sandbox & injecagent
    any_aux = agentdojo & (tool_sandbox | injecagent)

    indistinguishable = Counter()
    interface_rows = Counter()
    for row in rows:
        bridge_counts = Counter(
            str(catalog[candidate]["shared_action_ontology"]["bridge_key"])
            for candidate in row["legal_candidate_ids"]
        )
        interface_rows[str(row["source"])] += 1
        if any(count > 1 for count in bridge_counts.values()):
            indistinguishable[str(row["source"])] += 1
    residual_collisions = 0
    for row in rows:
        vectors = [
            ontology_candidate_vector(
                catalog[candidate],
                mode="ontology_local_residual",
                hash_dimension=128,
            ).tobytes()
            for candidate in row["legal_candidate_ids"]
        ]
        residual_collisions += int(len(vectors) != len(set(vectors)))

    checks = {
        "ontology_version_frozen": output["shared_action_ontology_version"]
        == ACTION_ONTOLOGY_VERSION,
        "all_candidates_annotated": all(
            "shared_action_ontology" in descriptor for descriptor in catalog.values()
        ),
        "all_rows_annotated": len(rows) == len(dataset["rows"]),
        "targets_unchanged": all(
            left["target_candidate_id"] == right["target_candidate_id"]
            for left, right in zip(dataset["rows"], rows)
        ),
        "legal_candidates_unchanged": all(
            left["legal_candidate_ids"] == right["legal_candidate_ids"]
            for left, right in zip(dataset["rows"], rows)
        ),
        "minimum_agentdojo_aux_bridge_keys": len(any_aux) >= 5,
        "minimum_all_three_bridge_keys": len(all_three) >= 2,
        "zero_local_residual_vector_collisions_within_interfaces": residual_collisions == 0,
    }
    audit = {
        "ontology_version": ACTION_ONTOLOGY_VERSION,
        "passed": all(checks.values()),
        "checks": checks,
        "candidate_count": len(catalog),
        "bridge_keys_by_source": {
            source: len(values) for source, values in sorted(source_bridges.items())
        },
        "agentdojo_bridge_keys_shared_with_any_auxiliary": len(any_aux),
        "bridge_keys_shared_by_all_three_sources": len(all_three),
        "shared_bridge_keys_all_three": sorted(all_three),
        "operations_by_source": {
            source: dict(sorted(values.items()))
            for source, values in sorted(source_operations.items())
        },
        "ontology_only_interface_collision_rows": dict(sorted(indistinguishable.items())),
        "interface_rows": dict(sorted(interface_rows.items())),
        "ontology_local_residual_collision_rows": residual_collisions,
        "dataset_content_sha256": stable_hash(output),
        "counterevidence": {
            "ontology_only_can_collapse_distinct_legal_candidates": bool(sum(indistinguishable.values())),
            "local_residual_is_required_for_within_source_identifiability": True,
            "mapping_uses_targets_or_outcomes": False,
        },
    }
    return output, audit
