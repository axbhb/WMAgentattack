from wmagentattack.evidence_ledger import (
    EvidenceLedger,
    ledger_length_features,
    ledger_text,
    parse_observation_facts,
    update_evidence_ledger,
)


def test_literal_and_colon_observations_are_parsed_deterministically():
    assert parse_observation_facts("{'Hotel A': {'price': 180}}") == (
        ("Hotel A.price", "180"),
    )
    rows = parse_observation_facts(
        "Airline: EasyJet, Flight Number: EJ789, Price: 150.0"
    )
    assert rows == (
        ("airline", "EasyJet"),
        ("flight_number", "EJ789"),
        ("price", "150.0"),
    )


def test_read_only_evidence_updates_memory_without_outcome_labels():
    ledger = update_evidence_ledger(
        EvidenceLedger(),
        goal="Find the cheapest Paris hotel and report its price",
        tool_name="get_hotels_prices",
        arguments={"hotel_names": ["Hotel A"]},
        observation_text="Hotel A: Price: 180",
        step_index=0,
        execution_status="success",
        error_type=None,
        argument_link_resolution={"unique": 1, "ambiguous": 0, "no_match": 0},
        state_changed=False,
    )
    assert ledger.items
    assert all(item.state_provenance == "read_only" for item in ledger.items)
    assert all(not item.outcome_labels_present for item in ledger.items)
    payload = ledger.model_dump(mode="json")
    assert "utility" not in str(payload).lower()
    assert "security" not in str(payload).lower()
    assert "Hotel A" in ledger_text(ledger)
    assert ledger_length_features(ledger)["item_count"] >= 1


def test_duplicate_conflict_and_error_are_explicit():
    first = update_evidence_ledger(
        EvidenceLedger(),
        goal="price",
        tool_name="price_tool",
        arguments={"name": "Hotel A"},
        observation_text="Hotel A: 180",
        step_index=0,
        execution_status="success",
        error_type=None,
        argument_link_resolution={"unique": 1},
        state_changed=False,
    )
    duplicate = update_evidence_ledger(
        first,
        goal="price",
        tool_name="price_tool",
        arguments={"name": "Hotel A"},
        observation_text="Hotel A: 180",
        step_index=1,
        execution_status="success",
        error_type=None,
        argument_link_resolution={"unique": 1},
        state_changed=False,
    )
    assert duplicate.items[-1].novelty == "duplicate"
    conflict = update_evidence_ledger(
        duplicate,
        goal="price",
        tool_name="price_tool",
        arguments={"name": "Hotel A"},
        observation_text="Hotel A: 200",
        step_index=2,
        execution_status="success",
        error_type=None,
        argument_link_resolution={"unique": 1},
        state_changed=False,
    )
    assert conflict.items[-1].conflict_status == "conflict"
    errored = update_evidence_ledger(
        conflict,
        goal="price",
        tool_name="price_tool",
        arguments={"name": "Hotel A"},
        observation_text="ignored raw error output",
        step_index=3,
        execution_status="error",
        error_type="ValidationError",
        argument_link_resolution={"ambiguous": 1},
        state_changed=False,
    )
    assert errored.items[-1].attribute == "execution_error"
    assert errored.items[-1].argument_link_status == "AMBIGUOUS"
