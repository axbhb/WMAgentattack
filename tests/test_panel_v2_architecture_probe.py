from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel

from wmagentattack.factorized_evaluator_v2 import (
    CallPattern,
    CanonicalExecutedCall,
    DifficultyFeatures,
    EvidenceObligation,
    EvidenceRoute,
    ProofContract,
    TaskDifficulty,
)
from wmagentattack.panel_v2_architecture_probe import (
    FROZEN_ARCHITECTURE_VARIANTS,
    EvidenceProgressStatus,
    CandidateDynamicsProbe,
    EvidenceProgressProbe,
    assess_obligation_progress,
    canonical_argument_key_target,
    feature_size,
    ledger_feature_payload,
    load_panel_v2_adapter_registry,
    prefix_feature_vector,
)
from wmagentattack.structured_ledger_v2 import StructuredEvidenceLedgerV2


ROOT = Path(__file__).resolve().parents[1]


def _prefix():
    return {
        "features": {
            "trusted_goal": "Find the matching record.",
            "last_action": {"function": "lookup", "arguments": {"id": "1"}},
            "legal_tools": ["lookup", "mutate"],
            "track": "deterministic_greedy",
            "prefix_index": 1,
            "last_observation": {"id": "1", "value": "x"},
            "execution_receipt": {"status": "success", "error_type": None},
            "causal_state_summary": {
                "last_state_changed": False,
                "cumulative_state_changes": 0,
                "cumulative_errors": 0,
                "last_delta_count": 0,
            },
            "ledger_v2": ledger_feature_payload(StructuredEvidenceLedgerV2()),
        }
    }


def test_multi_suite_adapter_extension_is_hash_bound_and_complete_for_new_modes():
    registry = load_panel_v2_adapter_registry(
        ROOT / "configs" / "0729_custom_panel_v2_adapter_extension.json"
    )
    assert registry.benchmark_version == "v1.2.2"
    assert "get_balance" in registry.adapters
    assert "get_hotels_prices" in registry.adapters
    assert registry.adapters["get_balance"].mode.value == "VALUE"
    assert registry.adapters["get_channels"].mode.value == "SCALAR_LIST"


def test_architecture_features_are_strictly_nested_and_deterministic():
    prefix = _prefix()
    sizes = []
    for variant in FROZEN_ARCHITECTURE_VARIANTS:
        first = prefix_feature_vector(prefix, variant=variant, hash_dimension=16)
        second = prefix_feature_vector(prefix, variant=variant, hash_dimension=16)
        assert np.array_equal(first, second)
        assert first.shape == (feature_size(variant, 16),)
        sizes.append(first.shape[0])
    assert sizes[0] < sizes[1] < sizes[2]


def test_outcome_or_future_fields_are_rejected_as_inputs():
    prefix = _prefix()
    prefix["features"]["utility"] = True
    try:
        prefix_feature_vector(prefix, variant="semantic_markov", hash_dimension=16)
    except ValueError as error:
        assert "leakage" in str(error)
    else:  # pragma: no cover
        raise AssertionError("outcome leakage was accepted")


class _Parameters(BaseModel):
    query: str


class _Tool:
    name = "lookup"
    parameters = _Parameters


class _OptionalParameters(BaseModel):
    required: str
    optional: int = 7


class _OptionalTool:
    name = "typed_lookup"
    parameters = _OptionalParameters


class _NoParameters(BaseModel):
    pass


class _NoArgumentTool:
    name = "list_all"
    parameters = _NoParameters


def test_argument_targets_use_post_validation_fields_and_drop_parser_extras():
    assert canonical_argument_key_target(
        _OptionalTool(), {"required": "x", "ignored": 1}
    ) == ("required",)
    assert canonical_argument_key_target(
        _NoArgumentTool(), {"properties": {}}
    ) == ()


def test_invalid_calls_keep_only_declared_attempted_argument_keys():
    assert canonical_argument_key_target(
        _OptionalTool(), {"optional": "not-an-int", "junk": True}
    ) == ("optional",)


def _contract():
    return ProofContract(
        contract_id="fixture",
        task_id="fixture",
        suite="fixture",
        task_difficulty=TaskDifficulty.L2,
        task_archetype="dependent_lookup",
        difficulty_features=DifficultyFeatures(
            required_tool_count=1,
            required_source_count=1,
            goal_atom_count=1,
            candidate_count=1,
            has_condition=False,
            has_mutation=False,
            requires_cross_source_join=False,
            requires_uniqueness_proof=False,
            required_report_slot_count=0,
        ),
        state_action_applicable=False,
        evidence_applicable=True,
        report_applicable=False,
        evidence_obligations=(
            EvidenceObligation(
                obligation_id="lookup_seen",
                description="The lookup was observed.",
                routes=(
                    EvidenceRoute(
                        route_id="lookup",
                        calls=(CallPattern(function="lookup", args={"query": "x"}),),
                    ),
                ),
            ),
        ),
    )


def _call(*, success=True):
    return CanonicalExecutedCall(
        call_index=0,
        function="lookup",
        raw_args={"query": "x"},
        canonical_args={"query": "x"},
        recorded_error=None if success else "ValueError",
        replay_error=None if success else "ValueError",
        executed_successfully=success,
        mutating=False,
    )


def test_obligation_progress_separates_unobserved_supported_and_contradicted():
    contract = _contract()
    empty = assess_obligation_progress([], contract, [_Tool()])
    supported = assess_obligation_progress([_call()], contract, [_Tool()])
    contradicted = assess_obligation_progress(
        [_call(success=False)], contract, [_Tool()]
    )
    assert empty[0]["status"] == EvidenceProgressStatus.UNOBSERVED.value
    assert supported[0]["status"] == EvidenceProgressStatus.SUPPORTED.value
    assert contradicted[0]["status"] == EvidenceProgressStatus.CONTRADICTED.value


def test_probe_heads_support_shared_candidate_scoring_and_four_evidence_states():
    dynamics = CandidateDynamicsProbe(
        prefix_size=17,
        candidate_size=8,
        argument_keys=5,
        hidden_size=12,
        dropout=0.0,
    )
    action, arguments = dynamics(torch.zeros(3, 17), torch.zeros(7, 8))
    assert action.shape == (3, 7)
    assert arguments.shape == (3, 5)
    evidence = EvidenceProgressProbe(input_size=25, hidden_size=12, dropout=0.0)
    assert evidence(torch.zeros(4, 25)).shape == (4, 4)
