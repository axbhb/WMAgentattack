from pathlib import Path

import pytest
from pydantic import ValidationError

from wmagentattack.goal_progress_v2 import (
    ComparisonOperator,
    CompletionObservation,
    GoalAtom,
    GoalAtomKind,
    GoalAtomPlan,
    GoalAtomStatus,
    ProgressType,
    assess_goal_progress,
    build_environment_fact,
    compile_goal_plan,
)
from wmagentattack.state_storage_v2 import VisibilityScope
from wmagentattack.structured_ledger_v2 import (
    AttributeKind,
    ExecutionChannelStatus,
    ItemAttributeInput,
    StructuredEvidenceLedgerV2,
    build_item_linkage_record,
    load_adapter_registry,
    update_structured_ledger,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = load_adapter_registry(
    ROOT / "configs" / "0726_travel_tool_output_adapter_registry_v2.json"
)


def _knowledge_plan(*, include_relation=True):
    atoms = [
        {
            "atom_id": "rating",
            "description": "candidate rating is at least 4.5",
            "progress_type": "KNOWLEDGE",
            "kind": "ATTRIBUTE_PREDICATE",
            "entity_type": "restaurant",
            "attribute_name": "rating",
            "operator": "GE",
            "target_value": 4.5,
        },
        {
            "atom_id": "price",
            "description": "candidate price is below 200",
            "progress_type": "KNOWLEDGE",
            "kind": "ATTRIBUTE_PREDICATE",
            "entity_type": "restaurant",
            "attribute_name": "price_per_person",
            "operator": "LT",
            "target_value": 200,
        },
    ]
    if include_relation:
        atoms.append(
            {
                "atom_id": "same",
                "description": "rating and price refer to the same candidate",
                "progress_type": "KNOWLEDGE",
                "kind": "SAME_ENTITY",
                "member_atom_ids": ["rating", "price"],
            }
        )
    return compile_goal_plan(
        task_id="fixture",
        trusted_goal="Find one restaurant rated at least 4.5 and priced below 200.",
        tool_schemas=[{"name": "rating"}, {"name": "price"}],
        atom_declarations=atoms,
    )


def _add(ledger, call_index, tool, arguments, output):
    return update_structured_ledger(
        ledger,
        REGISTRY,
        episode_id="goal-fixture",
        call_index=call_index,
        channel_status=ExecutionChannelStatus.EXECUTED_SUCCESS,
        tool_name=tool,
        arguments=arguments,
        runtime_output=output,
    ).ledger


def test_compiler_is_deterministic_and_rejects_invalid_relation_order():
    first = _knowledge_plan()
    second = _knowledge_plan()
    assert first == second
    assert first.trusted_goal_fingerprint == second.trusted_goal_fingerprint
    assert first.tool_schema_fingerprint == second.tool_schema_fingerprint
    with pytest.raises(ValidationError):
        GoalAtomPlan(
            task_id="bad",
            trusted_goal="goal",
            trusted_goal_fingerprint="0" * 64,
            tool_schema_fingerprint="1" * 64,
            atoms=(
                GoalAtom(
                    atom_id="same",
                    description="bad order",
                    progress_type=ProgressType.KNOWLEDGE,
                    kind=GoalAtomKind.SAME_ENTITY,
                    member_atom_ids=("later-a", "later-b"),
                ),
            ),
        )


def test_same_entity_goal_is_supported_only_when_constraints_share_entity():
    ledger = _add(
        StructuredEvidenceLedgerV2(),
        0,
        "get_rating_reviews_for_restaurants",
        {"restaurant_names": ["R1"]},
        {"R1": "Rating: 4.7\nReviews: Great"},
    )
    ledger = _add(
        ledger,
        1,
        "get_price_for_restaurants",
        {"restaurant_names": ["R1"]},
        {"R1": 180.0},
    )
    snapshot = assess_goal_progress(_knowledge_plan(), ledger)
    assert [row.status for row in snapshot.assessments] == [
        GoalAtomStatus.SUPPORTED,
        GoalAtomStatus.SUPPORTED,
        GoalAtomStatus.SUPPORTED,
    ]
    assert snapshot.coverage_by_type["KNOWLEDGE"] == 1.0
    assert not snapshot.completion_ready


def test_split_entity_support_is_not_mistaken_for_same_entity_completion():
    ledger = _add(
        StructuredEvidenceLedgerV2(),
        0,
        "get_rating_reviews_for_restaurants",
        {"restaurant_names": ["R1"]},
        {"R1": "Rating: 4.7\nReviews: Great"},
    )
    ledger = _add(
        ledger,
        1,
        "get_price_for_restaurants",
        {"restaurant_names": ["R2"]},
        {"R2": 180.0},
    )
    snapshot = assess_goal_progress(_knowledge_plan(), ledger)
    assert snapshot.assessments[0].status == GoalAtomStatus.SUPPORTED
    assert snapshot.assessments[1].status == GoalAtomStatus.SUPPORTED
    assert snapshot.assessments[2].status == GoalAtomStatus.CONTRADICTED
    assert snapshot.coverage_by_type["KNOWLEDGE"] == pytest.approx(2 / 3)


def test_item_local_ambiguous_and_unlinked_evidence_remain_distinct():
    plan = compile_goal_plan(
        task_id="linkage",
        trusted_goal="Find a restaurant with a known address.",
        tool_schemas=[{"name": "address"}],
        atom_declarations=[
            {
                "atom_id": "address",
                "description": "address is known",
                "progress_type": "KNOWLEDGE",
                "kind": "ATTRIBUTE_KNOWN",
                "entity_type": "restaurant",
                "attribute_name": "address",
            }
        ],
    )
    common = {
        "family": "restaurant",
        "entity_type": "restaurant",
        "episode_id": "linkage",
        "call_index": 0,
        "source_tool": "fixture",
        "source_arguments": {},
        "attributes": [
            ItemAttributeInput(
                name="address", value="1 Main St", kind=AttributeKind.SINGLE_VALUED
            )
        ],
    }
    ambiguous = build_item_linkage_record(
        **common,
        record_index=0,
        candidate_entity_keys=[{"name": "R1"}, {"name": "R2"}],
    )
    unlinked = build_item_linkage_record(**common, record_index=1)
    ambiguous_result = assess_goal_progress(
        plan, StructuredEvidenceLedgerV2(records=(ambiguous,))
    )
    unlinked_result = assess_goal_progress(
        plan, StructuredEvidenceLedgerV2(records=(unlinked,))
    )
    assert ambiguous_result.assessments[0].status == GoalAtomStatus.AMBIGUOUS
    assert unlinked_result.assessments[0].status == GoalAtomStatus.PARTIALLY_SUPPORTED
    assert ambiguous.provisional_entity_id != unlinked.provisional_entity_id


def test_environment_progress_uses_privileged_facts_without_entering_knowledge():
    plan = compile_goal_plan(
        task_id="environment",
        trusted_goal="Reserve Hotel A.",
        tool_schemas=[{"name": "reserve_hotel"}],
        atom_declarations=[
            {
                "atom_id": "reservation",
                "description": "hotel reservation is confirmed",
                "progress_type": "ENVIRONMENT",
                "kind": "ATTRIBUTE_PREDICATE",
                "entity_type": "hotel_reservation",
                "entity_key": {"hotel": "Hotel A"},
                "attribute_name": "status",
                "operator": "EQ",
                "target_value": "confirmed",
            }
        ],
    )
    fact = build_environment_fact(
        entity_type="hotel_reservation",
        entity_key={"hotel": "Hotel A"},
        attribute_name="status",
        value="confirmed",
        context={"start_day": "2026-08-01"},
        source_call_index=2,
    )
    snapshot = assess_goal_progress(
        plan, StructuredEvidenceLedgerV2(), environment_facts=[fact]
    )
    assert fact.visibility_scope == VisibilityScope.PLANNER_PRIVILEGED
    assert snapshot.assessments[0].status == GoalAtomStatus.SUPPORTED
    assert snapshot.coverage_by_type == {
        "KNOWLEDGE": 0.0,
        "ENVIRONMENT": 1.0,
        "COMPLETION": 0.0,
    }


def test_completion_progress_requires_answer_field_and_observed_stop():
    plan = compile_goal_plan(
        task_id="completion",
        trusted_goal="Tell me the selected restaurant name.",
        tool_schemas=[],
        atom_declarations=[
            {
                "atom_id": "answer_name",
                "description": "answer contains selected restaurant",
                "progress_type": "COMPLETION",
                "kind": "COMPLETION_FIELD",
                "attribute_name": "selected_restaurant",
                "operator": "KNOWN",
            }
        ],
    )
    not_stopped = CompletionObservation(
        stop_decision_observed=False,
        final_answer_present=True,
        answer_fields={"selected_restaurant": "R1"},
    )
    stopped = not_stopped.model_copy(update={"stop_decision_observed": True})
    before = assess_goal_progress(
        plan, StructuredEvidenceLedgerV2(), completion_observation=not_stopped
    )
    after = assess_goal_progress(
        plan, StructuredEvidenceLedgerV2(), completion_observation=stopped
    )
    assert before.assessments[0].status == GoalAtomStatus.SUPPORTED
    assert not before.completion_ready
    assert after.completion_ready


def test_incompatible_observed_value_is_contradicted_and_belief_is_one_hot():
    ledger = _add(
        StructuredEvidenceLedgerV2(),
        0,
        "get_price_for_restaurants",
        {"restaurant_names": ["R1"]},
        {"R1": 250.0},
    )
    snapshot = assess_goal_progress(
        _knowledge_plan(include_relation=False), ledger
    )
    rating, price = snapshot.assessments
    assert rating.status == GoalAtomStatus.UNSUPPORTED
    assert price.status == GoalAtomStatus.CONTRADICTED
    for row in snapshot.assessments:
        assert set(row.status_probabilities) == {status.value for status in GoalAtomStatus}
        assert sum(row.status_probabilities.values()) == 1.0
        assert row.status_probabilities[row.status.value] == 1.0
