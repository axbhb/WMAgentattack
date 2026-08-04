"""Causal representation and probe models for the custom clean panel-v2 ablation.

The three representations are strictly nested.  They consume only the trusted
goal, the policy track, executed calls, victim-visible tool observations, and a
label-blind StructuredEvidenceLedgerV2.  Factorized end labels, final reports,
expert calls, and future observations are never accepted as inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from agentdojo.functions_runtime import Function
from pydantic import TypeAdapter
from torch import Tensor, nn

from .clean_evidence_probe import hashed_text
from .decision_state import canonical_json_value
from .factorized_evaluator_v2 import (
    CanonicalExecutedCall,
    EvidenceObligation,
    ProofContract,
    call_matches_pattern,
)
from .structured_ledger_v2 import (
    AdapterRegistry,
    AdapterSpec,
    StructuredEvidenceLedgerV2,
    load_adapter_registry,
)


FROZEN_ARCHITECTURE_VARIANTS = (
    "semantic_markov",
    "observable_execution",
    "observable_execution_ledger_v2",
)


class EvidenceProgressStatus(str, Enum):
    UNOBSERVED = "UNOBSERVED"
    SUPPORTED = "SUPPORTED"
    AMBIGUOUS = "AMBIGUOUS"
    CONTRADICTED = "CONTRADICTED"


EVIDENCE_PROGRESS_STATUSES = tuple(status.value for status in EvidenceProgressStatus)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_panel_v2_adapter_registry(extension_path: Path) -> AdapterRegistry:
    """Load the frozen Travel registry plus reviewed multi-suite extensions."""

    payload = json.loads(extension_path.read_text(encoding="utf-8"))
    if payload.get("outcome_labels_present") is not False:
        raise ValueError("adapter extension must remain outcome-label blind")
    root = extension_path.resolve().parents[1]
    base_path = root / str(payload["base_registry"])
    if _sha256(base_path) != str(payload["base_registry_sha256"]):
        raise ValueError("base adapter registry hash mismatch")
    base = load_adapter_registry(base_path)
    additions = {
        str(name): AdapterSpec.model_validate(spec)
        for name, spec in payload["additional_adapters"].items()
    }
    overlap = sorted(set(base.adapters) & set(additions))
    if overlap:
        raise ValueError(f"adapter extension shadows base tools: {overlap}")
    return AdapterRegistry(
        schema_version=str(payload["schema_version"]),
        benchmark_version=str(payload["benchmark_version"]),
        suite="custom_panel_v2_multi_suite",
        adapters={**base.adapters, **additions},
    )


def canonical_argument_key_target(
    tool: Function,
    arguments: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return the post-validation argument fields represented by an action.

    AgentDojo's Pydantic input models may coerce values and ignore extra keys.
    The dynamics target must therefore describe the typed action that reaches
    the sandbox, rather than parser artifacts in the raw proposal.  For calls
    that fail validation, retain only raw keys that belong to the declared
    schema so error-recovery examples remain usable without expanding the head
    with arbitrary malformed keys.
    """

    schema_properties = {
        str(key)
        for key in tool.parameters.model_json_schema().get("properties", {})
    }
    try:
        validated = tool.parameters.model_validate(arguments)
    except Exception:
        return tuple(sorted(schema_properties.intersection(map(str, arguments))))
    supplied_fields = {
        str(key) for key in getattr(validated, "model_fields_set", set())
    }
    return tuple(sorted(schema_properties.intersection(supplied_fields)))


def _without_runtime_ids(value: Any) -> Any:
    """Remove episode-local IDs that could become trajectory fingerprints."""

    if isinstance(value, Mapping):
        return {
            str(key): _without_runtime_ids(item)
            for key, item in value.items()
            if key
            not in {
                "record_id",
                "fact_id",
                "conflict_id",
                "episode_id",
                "arguments_fingerprint",
                "observation_fingerprint",
                "resolved_entity_id",
                "provisional_entity_id",
                "entity_id",
                "left_fact_id",
                "right_fact_id",
            }
        }
    if isinstance(value, list):
        return [_without_runtime_ids(item) for item in value]
    return value


def ledger_feature_payload(ledger: StructuredEvidenceLedgerV2) -> dict[str, Any]:
    records = []
    for record in sorted(ledger.records, key=lambda row: (row.call_index, row.record_index)):
        records.append(
            {
                "entity_type": record.entity_type,
                "entity_key": canonical_json_value(record.entity_key),
                "entity_candidates": [
                    canonical_json_value(candidate.entity_key)
                    for candidate in record.entity_candidates
                ],
                "link_status": record.link_status,
                "attributes": [
                    {
                        "name": attribute.name,
                        "value": canonical_json_value(attribute.value),
                        "kind": attribute.kind.value,
                    }
                    for attribute in record.attributes
                ],
                "context": canonical_json_value(record.context),
                "source_tool": record.source_tool,
                "source_arguments": canonical_json_value(record.source_arguments),
                "call_index": record.call_index,
                "execution_status": record.execution_status,
                "state_provenance": record.state_provenance,
            }
        )
    conflicts = [
        {
            "attribute_name": row.attribute_name,
            "reason": row.reason,
        }
        for row in ledger.conflicts
    ]
    payload = {
        "records": records,
        "conflicts": conflicts,
        "execution_receipts": [
            {
                "call_index": row.call_index,
                "tool_name": row.tool_name,
                "execution_status": row.execution_status,
            }
            for row in ledger.execution_receipts
        ],
    }
    return _without_runtime_ids(payload)


def _log_count(value: Any) -> float:
    return math.log1p(float(value or 0.0))


def prefix_feature_vector(
    prefix: Mapping[str, Any], *, variant: str, hash_dimension: int
) -> np.ndarray:
    if variant not in FROZEN_ARCHITECTURE_VARIANTS:
        raise ValueError(f"unknown architecture variant: {variant}")
    features = prefix["features"]
    forbidden = {
        "factorized",
        "utility",
        "security",
        "final_report",
        "final_output",
        "expert_calls",
        "future_calls",
    }
    leaked = sorted(forbidden & set(features))
    if leaked:
        raise ValueError(f"outcome/future feature leakage: {leaked}")

    goal = hashed_text(features["trusted_goal"], hash_dimension, "panel-v2-goal")
    last_action = hashed_text(
        features["last_action"], hash_dimension, "panel-v2-last-action"
    )
    legal = hashed_text(
        sorted(features["legal_tools"]), hash_dimension, "panel-v2-legal-tools"
    )
    policy = hashed_text(
        {"track": features["track"]}, hash_dimension, "panel-v2-policy"
    )
    parts: list[np.ndarray] = [
        goal,
        last_action,
        legal,
        policy,
        np.asarray([_log_count(features["prefix_index"])], dtype=np.float32),
    ]
    if variant != "semantic_markov":
        observation = hashed_text(
            features["last_observation"],
            hash_dimension,
            "panel-v2-observation",
        )
        execution = hashed_text(
            features["execution_receipt"],
            hash_dimension,
            "panel-v2-execution",
        )
        state = hashed_text(
            features["causal_state_summary"],
            hash_dimension,
            "panel-v2-causal-state",
        )
        state_summary = features["causal_state_summary"]
        parts.extend(
            [
                observation,
                execution,
                state,
                np.asarray(
                    [
                        float(bool(state_summary.get("last_state_changed", False))),
                        _log_count(state_summary.get("cumulative_state_changes", 0)),
                        _log_count(state_summary.get("cumulative_errors", 0)),
                        _log_count(state_summary.get("last_delta_count", 0)),
                    ],
                    dtype=np.float32,
                ),
            ]
        )
    if variant == "observable_execution_ledger_v2":
        ledger_payload = features["ledger_v2"]
        ledger = hashed_text(
            ledger_payload, hash_dimension, "panel-v2-ledger-v2"
        )
        parts.extend(
            [
                ledger,
                goal * ledger,
                np.asarray(
                    [
                        _log_count(len(ledger_payload.get("records", []))),
                        _log_count(len(ledger_payload.get("conflicts", []))),
                        _log_count(
                            len(ledger_payload.get("execution_receipts", []))
                        ),
                    ],
                    dtype=np.float32,
                ),
            ]
        )
    return np.concatenate(parts).astype(np.float32, copy=False)


def candidate_descriptor_vector(
    descriptor: Mapping[str, Any], *, hash_dimension: int
) -> np.ndarray:
    # Share the goal namespace so lexical overlap with previously unseen tool
    # names/descriptions remains available to the candidate scorer.
    return hashed_text(descriptor, hash_dimension, "panel-v2-goal")


def obligation_descriptor_vector(
    prefix: Mapping[str, Any],
    obligation: Mapping[str, Any],
    *,
    hash_dimension: int,
) -> np.ndarray:
    goal = hashed_text(
        prefix["features"]["trusted_goal"], hash_dimension, "panel-v2-goal"
    )
    descriptor = hashed_text(
        {
            "obligation_id": obligation["obligation_id"],
            "description": obligation["description"],
        },
        hash_dimension,
        "panel-v2-goal",
    )
    return np.concatenate((descriptor, goal * descriptor)).astype(
        np.float32, copy=False
    )


def _matched_pattern_count(
    calls: Sequence[CanonicalExecutedCall],
    obligation: EvidenceObligation,
    tools: Mapping[str, Function],
    *,
    first_successful_mutation: int | None,
) -> tuple[bool, bool, bool]:
    full = False
    partial = False
    contradicted = False
    required_functions = {
        pattern.function for route in obligation.routes for pattern in route.calls
    }
    if any(
        call.function in required_functions and not call.executed_successfully
        for call in calls
    ):
        contradicted = True

    for route in obligation.routes:
        eligible = [
            call
            for call in calls
            if not route.must_precede_first_mutation
            or first_successful_mutation is None
            or call.call_index < first_successful_mutation
        ]
        unused = set(range(len(eligible)))
        matched = 0
        for pattern in route.calls:
            found = next(
                (
                    index
                    for index in sorted(unused)
                    if call_matches_pattern(eligible[index], pattern, tools)
                ),
                None,
            )
            if found is not None:
                unused.remove(found)
                matched += 1
        if matched == len(route.calls):
            full = True
        elif matched:
            partial = True

        if route.must_precede_first_mutation and first_successful_mutation is not None:
            all_unused = set(range(len(calls)))
            matched_without_order = 0
            for pattern in route.calls:
                found = next(
                    (
                        index
                        for index in sorted(all_unused)
                        if call_matches_pattern(calls[index], pattern, tools)
                    ),
                    None,
                )
                if found is not None:
                    all_unused.remove(found)
                    matched_without_order += 1
            if matched_without_order == len(route.calls) and not full:
                contradicted = True
    return full, partial, contradicted


def assess_obligation_progress(
    calls: Sequence[CanonicalExecutedCall],
    contract: ProofContract,
    tools: Sequence[Function],
) -> list[dict[str, str]]:
    """Create target-only prefix labels without reading final outcome labels."""

    tool_map = {tool.name: tool for tool in tools}
    first_mutation = min(
        (
            call.call_index
            for call in calls
            if call.mutating and call.executed_successfully
        ),
        default=None,
    )
    rows = []
    for obligation in contract.evidence_obligations:
        full, partial, contradicted = _matched_pattern_count(
            calls,
            obligation,
            tool_map,
            first_successful_mutation=first_mutation,
        )
        if full:
            status = EvidenceProgressStatus.SUPPORTED
        elif contradicted:
            status = EvidenceProgressStatus.CONTRADICTED
        elif partial:
            status = EvidenceProgressStatus.AMBIGUOUS
        else:
            status = EvidenceProgressStatus.UNOBSERVED
        rows.append(
            {
                "obligation_id": obligation.obligation_id,
                "description": obligation.description,
                "status": status.value,
            }
        )
    return rows


class CandidateDynamicsProbe(nn.Module):
    def __init__(
        self,
        *,
        prefix_size: int,
        candidate_size: int,
        argument_keys: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.prefix_encoder = nn.Sequential(
            nn.Linear(prefix_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        self.score = nn.Linear(hidden_size, 1)
        self.argument_head = nn.Linear(hidden_size, argument_keys)

    def forward(
        self, prefixes: Tensor, candidates: Tensor
    ) -> tuple[Tensor, Tensor]:
        prefix = self.prefix_encoder(prefixes)
        candidate = self.candidate_encoder(candidates)
        joint = torch.tanh(prefix[:, None, :] + candidate[None, :, :])
        return self.score(joint).squeeze(-1), self.argument_head(prefix)


class EvidenceProgressProbe(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, len(EVIDENCE_PROGRESS_STATUSES)),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


def canonical_executed_call(
    *,
    call_index: int,
    function: str,
    arguments: Mapping[str, Any],
    error: str | None,
    tools: Mapping[str, Function],
    mutating_tools: set[str],
) -> CanonicalExecutedCall:
    tool = tools.get(function)
    canonical_args = canonical_json_value(dict(arguments))
    replay_error = error
    if tool is None:
        replay_error = replay_error or f"ToolNotFoundError: {function}"
    else:
        try:
            canonical_args = canonical_json_value(
                tool.parameters.model_validate(arguments).model_dump()
            )
        except Exception as exc:  # pragma: no cover - exercised by remote traces
            replay_error = replay_error or f"{type(exc).__name__}: {exc}"
    return CanonicalExecutedCall(
        call_index=call_index,
        function=function,
        raw_args=canonical_json_value(dict(arguments)),
        canonical_args=canonical_args,
        recorded_error=error,
        replay_error=replay_error,
        executed_successfully=error is None and replay_error is None,
        mutating=function in mutating_tools,
    )


def feature_size(variant: str, hash_dimension: int) -> int:
    fixture = {
        "features": {
            "trusted_goal": "fixture",
            "last_action": {"function": "<START>", "arguments": {}},
            "legal_tools": ["fixture"],
            "track": "deterministic_greedy",
            "prefix_index": 0,
            "last_observation": "",
            "execution_receipt": {"status": "start"},
            "causal_state_summary": {},
            "ledger_v2": {
                "records": [],
                "conflicts": [],
                "execution_receipts": [],
            },
        }
    }
    return int(
        prefix_feature_vector(
            fixture, variant=variant, hash_dimension=hash_dimension
        ).shape[0]
    )
