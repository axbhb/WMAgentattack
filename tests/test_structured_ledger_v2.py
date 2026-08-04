import json
from pathlib import Path

import pytest

import wmagentattack.structured_ledger_v2 as ledger_module
from wmagentattack.clean_state_instrumentation import canonical_call_signature
from wmagentattack.state_storage_v2 import VisibilityScope
from wmagentattack.structured_ledger_v2 import (
    AdapterMode,
    AdapterSpec,
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


def _update(ledger, *, call_index, tool, arguments, output, status=ExecutionChannelStatus.EXECUTED_SUCCESS):
    return update_structured_ledger(
        ledger,
        REGISTRY,
        episode_id="episode-fixture",
        call_index=call_index,
        channel_status=status,
        tool_name=tool,
        arguments=arguments,
        runtime_output=output,
        proposal_signature=canonical_call_signature(tool, arguments),
    )


def test_registry_covers_all_28_travel_tools():
    assert REGISTRY.benchmark_version == "v1.2.2"
    assert REGISTRY.suite == "travel"
    assert len(REGISTRY.adapters) == 28
    assert set(spec.mode for spec in REGISTRY.adapters.values()) == {
        AdapterMode.USER_FIELDS,
        AdapterMode.NAME_LIST_TEXT,
        AdapterMode.ENTITY_MAP,
        AdapterMode.PRICE_RANGE_MAP,
        AdapterMode.RATING_REVIEWS_MAP,
        AdapterMode.FLIGHT_LINES,
        AdapterMode.OBJECT,
        AdapterMode.OBJECT_LIST,
        AdapterMode.MUTATION_ACK,
    }


def test_value_adapter_preserves_typed_scalar_and_entity_context():
    registry = ledger_module.AdapterRegistry(
        schema_version="fixture",
        benchmark_version="v1.2.2",
        suite="fixture",
        adapters={
            "get_balance": AdapterSpec(
                family="banking",
                mode=AdapterMode.VALUE,
                entity_type="bank_account",
                fixed_entity_key={"role": "current_account"},
                attribute_name="balance",
            )
        },
    )
    result = update_structured_ledger(
        StructuredEvidenceLedgerV2(),
        registry,
        episode_id="value-fixture",
        call_index=0,
        channel_status=ExecutionChannelStatus.EXECUTED_SUCCESS,
        tool_name="get_balance",
        arguments={},
        runtime_output=1810.0,
    )
    record = result.ledger.records[0]
    assert record.entity_key == {"role": "current_account"}
    assert record.attributes[0].name == "balance"
    assert record.attributes[0].value == 1810.0


def test_scalar_list_adapter_keeps_item_boundaries_and_call_context():
    registry = ledger_module.AdapterRegistry(
        schema_version="fixture",
        benchmark_version="v1.2.2",
        suite="fixture",
        adapters={
            "get_users": AdapterSpec(
                family="slack",
                mode=AdapterMode.SCALAR_LIST,
                entity_type="slack_user",
                attribute_name="name",
                context_argument_fields=("channel",),
            )
        },
    )
    result = update_structured_ledger(
        StructuredEvidenceLedgerV2(),
        registry,
        episode_id="list-fixture",
        call_index=0,
        channel_status=ExecutionChannelStatus.EXECUTED_SUCCESS,
        tool_name="get_users",
        arguments={"channel": "general"},
        runtime_output=["Alice", "Bob"],
    )
    assert result.added_records == 2
    assert {row.entity_key["name"] for row in result.ledger.records} == {
        "Alice",
        "Bob",
    }
    assert all(row.context == {"channel": "general"} for row in result.ledger.records)


def test_price_map_preserves_hotel_record_boundaries_without_false_conflict():
    result = _update(
        StructuredEvidenceLedgerV2(),
        call_index=0,
        tool="get_hotels_prices",
        arguments={"hotel_names": ["Hotel A", "Hotel B"]},
        output={"Hotel A": "Price range: 180.0 - 200.0", "Hotel B": "Price range: 220.0 - 250.0"},
    )
    assert result.added_records == 2
    assert result.added_conflicts == 0
    assert {record.entity_key["name"] for record in result.ledger.records} == {
        "Hotel A",
        "Hotel B",
    }
    assert len({record.resolved_entity_id for record in result.ledger.records}) == 2


def test_same_entity_attribute_context_incompatible_value_creates_conflict():
    first = _update(
        StructuredEvidenceLedgerV2(),
        call_index=0,
        tool="get_price_for_restaurants",
        arguments={"restaurant_names": ["R1"]},
        output={"R1": 50.0},
    ).ledger
    second = _update(
        first,
        call_index=1,
        tool="get_price_for_restaurants",
        arguments={"restaurant_names": ["R1"]},
        output={"R1": 80.0},
    )
    assert second.added_conflicts == 1
    assert second.ledger.conflicts[0].attribute_name == "price_per_person"


def test_set_valued_differences_do_not_conflict():
    first = _update(
        StructuredEvidenceLedgerV2(),
        call_index=0,
        tool="get_car_types_available",
        arguments={"company_name": ["C1"]},
        output={"C1": ["SUV", "Sedan"]},
    ).ledger
    second = _update(
        first,
        call_index=1,
        tool="get_car_types_available",
        arguments={"company_name": ["C1"]},
        output={"C1": ["Truck", "SUV"]},
    )
    assert second.added_conflicts == 0


def test_unlinked_records_get_local_ids_and_never_conflict():
    ledger = StructuredEvidenceLedgerV2()
    for index, error in enumerate(("ValidationError", "RuntimeError")):
        result = update_structured_ledger(
            ledger,
            REGISTRY,
            episode_id="episode-fixture",
            call_index=index,
            channel_status=ExecutionChannelStatus.EXECUTED_ERROR,
            tool_name="get_hotels_prices",
            arguments={"hotel_names": ["Unknown"]},
            error_type=error,
        )
        ledger = result.ledger
    assert all(record.link_status == "UNLINKED" for record in ledger.records)
    assert len({record.provisional_entity_id for record in ledger.records}) == 2
    assert ledger.conflicts == ()


def test_ambiguous_record_preserves_all_candidates_without_forcing_resolution():
    spec = AdapterSpec(
        family="fixture",
        mode=AdapterMode.ENTITY_MAP,
        entity_type="hotel",
        attribute_name="price",
        attribute_kind=AttributeKind.SINGLE_VALUED,
    )
    draft = ledger_module._RecordDraft(
        candidate_entity_keys=({"name": "Hotel A"}, {"name": "Hotel B"}),
        attributes=(
            ledger_module._AttributeDraft(
                name="price", value=180, kind=AttributeKind.SINGLE_VALUED
            ),
        ),
    )
    record = ledger_module._build_record(
        spec=spec,
        draft=draft,
        episode_id="episode-fixture",
        call_index=0,
        record_index=0,
        tool_name="fixture",
        arguments={},
        execution_status="success",
        state_changed=False,
    )
    assert record.link_status == "AMBIGUOUS"
    assert record.resolved_entity_id is None
    assert record.provisional_entity_id is not None
    assert len(record.entity_candidates) == 2


def test_public_item_linker_handles_unique_ambiguous_and_unlinked_per_record():
    common = {
        "family": "hotel",
        "entity_type": "hotel",
        "episode_id": "linkage-fixture",
        "call_index": 3,
        "source_tool": "fixture",
        "source_arguments": {},
        "attributes": [
            ItemAttributeInput(
                name="price", value=100, kind=AttributeKind.SINGLE_VALUED
            )
        ],
    }
    unique = build_item_linkage_record(
        **common,
        record_index=0,
        candidate_entity_keys=[{"name": "A"}],
    )
    ambiguous = build_item_linkage_record(
        **common,
        record_index=1,
        candidate_entity_keys=[{"name": "A"}, {"name": "B"}],
    )
    unlinked = build_item_linkage_record(**common, record_index=2)
    assert unique.link_status == "UNIQUE"
    assert unique.entity_key == {"name": "A"}
    assert unique.resolved_entity_id is not None
    assert ambiguous.link_status == "AMBIGUOUS"
    assert {row.entity_key["name"] for row in ambiguous.entity_candidates} == {"A", "B"}
    assert ambiguous.provisional_entity_id is not None
    assert unlinked.link_status == "UNLINKED"
    assert unlinked.entity_candidates == ()
    assert unlinked.provisional_entity_id is not None
    assert ambiguous.provisional_entity_id != unlinked.provisional_entity_id


def test_item_linker_deduplicates_candidates_and_rejects_mixed_link_sources():
    attributes = [
        ItemAttributeInput(name="address", value="X", kind=AttributeKind.SINGLE_VALUED)
    ]
    record = build_item_linkage_record(
        family="hotel",
        entity_type="hotel",
        episode_id="dedupe",
        call_index=0,
        record_index=0,
        source_tool="fixture",
        source_arguments={},
        attributes=attributes,
        candidate_entity_keys=[{"name": "A"}, {"name": "A"}],
    )
    assert record.link_status == "UNIQUE"
    assert len(record.entity_candidates) == 1
    with pytest.raises(ValueError):
        build_item_linkage_record(
            family="hotel",
            entity_type="hotel",
            episode_id="dedupe",
            call_index=0,
            record_index=1,
            source_tool="fixture",
            source_arguments={},
            attributes=attributes,
            entity_key={"name": "A"},
            candidate_entity_keys=[{"name": "A"}],
        )


@pytest.mark.parametrize(
    "status",
    [
        ExecutionChannelStatus.PROPOSED,
        ExecutionChannelStatus.TERMINAL_UNEXECUTED,
        ExecutionChannelStatus.CENSORED,
    ],
)
def test_nonexecuted_channel_states_never_update_ledger(status):
    result = _update(
        StructuredEvidenceLedgerV2(),
        call_index=0,
        tool="get_hotels_prices",
        arguments={"hotel_names": ["Hotel A"]},
        output={"Hotel A": "Price range: 1 - 2"},
        status=status,
    )
    assert result.ledger == StructuredEvidenceLedgerV2()
    assert result.added_records == 0
    assert result.ignored_without_observation


def test_same_executed_record_replay_is_idempotent_and_mismatch_fails_closed():
    kwargs = {
        "call_index": 0,
        "tool": "get_hotels_address",
        "arguments": {"hotel_name": "Hotel A"},
        "output": {"Hotel A": "1 Main Street"},
    }
    first = _update(StructuredEvidenceLedgerV2(), **kwargs)
    replay = _update(first.ledger, **kwargs)
    assert replay.ledger == first.ledger
    assert replay.added_records == 0
    with pytest.raises(ValueError):
        _update(first.ledger, **{**kwargs, "output": {"Hotel A": "2 Main Street"}})
    with pytest.raises(ValueError):
        update_structured_ledger(
            StructuredEvidenceLedgerV2(),
            REGISTRY,
            episode_id="episode-fixture",
            call_index=0,
            channel_status=ExecutionChannelStatus.EXECUTED_SUCCESS,
            tool_name="get_hotels_address",
            arguments={"hotel_name": "Hotel A"},
            runtime_output={"Hotel A": "1 Main Street"},
            proposal_signature="wrong-signature",
        )


def test_flight_lines_become_distinct_entity_records():
    output = (
        "Airline: A, Flight Number: A1, Departure Time: 2026-01-01 10:00:00, "
        "Arrival Time: 2026-01-01 11:00:00, Price: 100.0, Contact Information: +1\n"
        "Airline: B, Flight Number: B2, Departure Time: 2026-01-01 12:00:00, "
        "Arrival Time: 2026-01-01 13:00:00, Price: 120.0, Contact Information: +2"
    )
    result = _update(
        StructuredEvidenceLedgerV2(),
        call_index=0,
        tool="get_flight_information",
        arguments={"departure_city": "Paris", "arrival_city": "London"},
        output=output,
    )
    assert result.added_records == 2
    assert {record.entity_key["flight_number"] for record in result.ledger.records} == {
        "A1",
        "B2",
    }


def test_object_records_and_scope_are_structured_and_label_blind():
    result = _update(
        StructuredEvidenceLedgerV2(),
        call_index=0,
        tool="create_calendar_event",
        arguments={
            "title": "Dinner",
            "start_time": "2026-01-01 18:00",
            "end_time": "2026-01-01 20:00",
        },
        output={
            "id_": "event-1",
            "title": "Dinner",
            "description": "",
            "start_time": "2026-01-01T18:00:00",
            "end_time": "2026-01-01T20:00:00",
            "location": None,
            "participants": ["user@example.com"],
            "all_day": False,
            "status": "confirmed",
        },
    )
    record = result.ledger.records[0]
    assert record.entity_key == {"id_": "event-1"}
    assert record.observation_scope == VisibilityScope.VICTIM_OBSERVED
    serialized = json.dumps(result.model_dump(mode="json")).lower()
    assert '"utility"' not in serialized
    assert '"security"' not in serialized
    assert '"expert' not in serialized
