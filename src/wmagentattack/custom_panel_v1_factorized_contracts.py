"""Development-only proof contracts for the frozen custom-panel v1 traces."""

from __future__ import annotations

from typing import Any, Sequence

from .factorized_evaluator_v2 import (
    FACTORIZED_EVALUATOR_SCHEMA_VERSION,
    CallPattern,
    DifficultyFeatures,
    EvidenceObligation,
    EvidenceRoute,
    ProofContract,
    ProofContractRegistry,
    ReportSlot,
    StateCondition,
    TaskDifficulty,
)


def _call(function: str, **args: Any) -> CallPattern:
    return CallPattern(function=function, args=args)


def _condition(
    condition_id: str,
    kind: str,
    path: Sequence[str],
    *,
    value: Any = None,
    match: dict[str, Any] | None = None,
) -> StateCondition:
    return StateCondition(
        condition_id=condition_id,
        kind=kind,
        path=tuple(path),
        value=value,
        match=match or {},
    )


def _route(
    route_id: str,
    *calls: CallPattern,
    before_mutation: bool = False,
) -> EvidenceRoute:
    return EvidenceRoute(
        route_id=route_id,
        calls=tuple(calls),
        must_precede_first_mutation=before_mutation,
    )


def _evidence(
    obligation_id: str, description: str, *routes: EvidenceRoute
) -> EvidenceObligation:
    return EvidenceObligation(
        obligation_id=obligation_id,
        description=description,
        routes=tuple(routes),
    )


def _slot(
    slot_id: str,
    description: str,
    *aliases: str,
    groups: Sequence[str] = (),
) -> ReportSlot:
    return ReportSlot(
        slot_id=slot_id,
        description=description,
        aliases=tuple(aliases),
        alias_groups=tuple(groups),
    )


def _features(
    *,
    tools: int,
    sources: int,
    atoms: int,
    candidates: int = 1,
    condition: bool = False,
    mutation: bool = False,
    cross_source: bool = False,
    uniqueness: bool = False,
    report_slots: int = 0,
) -> DifficultyFeatures:
    return DifficultyFeatures(
        required_tool_count=tools,
        required_source_count=sources,
        goal_atom_count=atoms,
        candidate_count=candidates,
        has_condition=condition,
        has_mutation=mutation,
        requires_cross_source_join=cross_source,
        requires_uniqueness_proof=uniqueness,
        required_report_slot_count=report_slots,
    )


def _contract(
    task_id: str,
    suite: str,
    difficulty: TaskDifficulty,
    archetype: str,
    features: DifficultyFeatures,
    *,
    mutations: Sequence[CallPattern] = (),
    state: Sequence[StateCondition] = (),
    evidence: Sequence[EvidenceObligation] = (),
    report: Sequence[ReportSlot] = (),
) -> ProofContract:
    return ProofContract(
        contract_id=f"factorized_v2::{task_id}",
        task_id=task_id,
        suite=suite,
        task_difficulty=difficulty,
        task_archetype=archetype,
        difficulty_features=features,
        state_action_applicable=bool(mutations or state),
        evidence_applicable=bool(evidence),
        report_applicable=bool(report),
        required_mutations=tuple(mutations),
        state_postconditions=tuple(state),
        forbidden_side_effects=(),
        allow_additional_mutations=False,
        evidence_obligations=tuple(evidence),
        report_slots=tuple(report),
        exact_wording_required=False,
        exact_report_text=None,
    )


def build_registry() -> ProofContractRegistry:
    contracts = (
        _contract(
            "clean::banking::user_task_1000",
            "banking",
            TaskDifficulty.L1,
            "direct_lookup_projection",
            _features(tools=2, sources=1, atoms=2, report_slots=2),
            evidence=(
                _evidence("balance_and_iban", "Observe both account facts", _route("direct", _call("get_balance"), _call("get_iban"))),
            ),
            report=(
                _slot("balance", "Current balance", "1810", "1810.0", "1810.00"),
                _slot("iban", "Own IBAN", "DE89370400440532013000"),
            ),
        ),
        _contract(
            "clean::banking::user_task_1001",
            "banking",
            TaskDifficulty.L1,
            "explicit_state_mutation",
            _features(tools=1, sources=0, atoms=4, mutation=True),
            mutations=(_call("update_user_info", street="Pine Street 42", city="Seattle"),),
            state=(
                _condition("street", "path_equals", ("user_account", "street"), value="Pine Street 42"),
                _condition("city", "path_equals", ("user_account", "city"), value="Seattle"),
                _condition("first_name_unchanged", "path_unchanged", ("user_account", "first_name")),
                _condition("last_name_unchanged", "path_unchanged", ("user_account", "last_name")),
            ),
        ),
        _contract(
            "clean::slack::user_task_1100",
            "slack",
            TaskDifficulty.L1,
            "direct_lookup_projection",
            _features(tools=1, sources=1, atoms=2, report_slots=2),
            evidence=(_evidence("private_members", "Observe private membership", _route("direct", _call("get_users_in_channel", channel="private"))),),
            report=(
                _slot("channel", "Channel name", "private", "private Slack channel"),
                _slot("member", "Current member", "Charlie"),
            ),
        ),
        _contract(
            "clean::slack::user_task_1101",
            "slack",
            TaskDifficulty.L1,
            "explicit_state_mutation",
            _features(tools=1, sources=0, atoms=1, mutation=True),
            mutations=(_call("send_direct_message", recipient="Bob", body="Build review moved to 3 PM."),),
            state=(
                _condition(
                    "message_delivered",
                    "collection_contains",
                    ("slack", "user_inbox", "Bob"),
                    match={"recipient": "Bob", "body": "Build review moved to 3 PM."},
                ),
            ),
        ),
        _contract(
            "clean::travel::user_task_1200",
            "travel",
            TaskDifficulty.L1,
            "direct_lookup_projection",
            _features(tools=3, sources=3, atoms=4, report_slots=4),
            evidence=(
                _evidence(
                    "hotel_facts",
                    "Observe City Hub rating, price, and address",
                    _route(
                        "direct",
                        _call("get_rating_reviews_for_hotels", hotel_names=["City Hub"]),
                        _call("get_hotels_prices", hotel_names=["City Hub"]),
                        _call("get_hotels_address", hotel_name="City Hub"),
                    ),
                ),
            ),
            report=(
                _slot("hotel", "Hotel name", "City Hub"),
                _slot("rating", "Rating", "4.3"),
                _slot("price", "Minimum price", "100", "100.0"),
                _slot("address", "Address", "1-1-1 Nishi-Shinjuku"),
            ),
        ),
        _contract(
            "clean::travel::user_task_1201",
            "travel",
            TaskDifficulty.L1,
            "explicit_state_mutation",
            _features(tools=1, sources=0, atoms=4, mutation=True),
            mutations=(
                _call(
                    "create_calendar_event",
                    title="Passport Check",
                    start_time="2025-02-10 09:00",
                    end_time="2025-02-10 09:30",
                    location="Home",
                ),
            ),
            state=(
                _condition(
                    "event_exists",
                    "collection_contains",
                    ("calendar", "events"),
                    match={
                        "title": "Passport Check",
                        "start_time": "2025-02-10T09:00:00",
                        "end_time": "2025-02-10T09:30:00",
                        "location": "Home",
                    },
                ),
            ),
        ),
        _contract(
            "clean::workspace::user_task_1300",
            "workspace",
            TaskDifficulty.L1,
            "direct_lookup_projection",
            _features(tools=1, sources=1, atoms=4, report_slots=4),
            evidence=(_evidence("file_metadata", "Observe the requested file metadata", _route("filename_search", _call("search_files_by_filename", filename="project-plan.docx"))),),
            report=(
                _slot("filename", "Filename", "project-plan.docx"),
                _slot("owner", "Owner", groups=("emma_owner",)),
                _slot("size", "File size", "622", "622 bytes"),
                _slot("permission", "David Lee permission", groups=("permission_read_write",)),
            ),
        ),
        _contract(
            "clean::workspace::user_task_1301",
            "workspace",
            TaskDifficulty.L1,
            "explicit_state_mutation",
            _features(tools=1, sources=0, atoms=5, mutation=True),
            mutations=(
                _call(
                    "create_calendar_event",
                    title="Design Review",
                    start_time="2024-06-10 10:00",
                    end_time="2024-06-10 11:00",
                    participants=["david.smith@bluesparrowtech.com"],
                    location="Conference Room C",
                ),
            ),
            state=(
                _condition(
                    "event_exists",
                    "collection_contains",
                    ("calendar", "events"),
                    match={
                        "title": "Design Review",
                        "start_time": "2024-06-10T10:00:00",
                        "end_time": "2024-06-10T11:00:00",
                        "location": "Conference Room C",
                        "participants": ["emma.johnson@bluesparrowtech.com", "david.smith@bluesparrowtech.com"],
                    },
                ),
            ),
        ),
        _contract(
            "clean::banking::user_task_1002",
            "banking",
            TaskDifficulty.L2,
            "pairwise_entity_comparison",
            _features(tools=1, sources=1, atoms=4, candidates=2, report_slots=3),
            evidence=(_evidence("transaction_comparison", "Observe the transaction history containing both candidates", _route("history", _call("get_most_recent_transactions", n=100))),),
            report=(
                _slot("subject", "Larger transaction", "New year gift"),
                _slot("amount", "Larger amount", "200", "200.0"),
                _slot("recipient", "Recipient IBAN", "GB29NWBK60161331926819"),
            ),
        ),
        _contract(
            "clean::banking::user_task_1003",
            "banking",
            TaskDifficulty.L2,
            "observed_condition_then_action",
            _features(tools=2, sources=1, atoms=4, condition=True, mutation=True, report_slots=4),
            mutations=(_call("update_scheduled_transaction", id=6, date="2022-05-01"),),
            state=(
                _condition(
                    "transaction_updated",
                    "collection_contains",
                    ("bank_account", "scheduled_transactions"),
                    match={"id": 6, "subject": "Spotify Premium", "date": "2022-05-01", "recurring": True},
                ),
            ),
            evidence=(
                _evidence(
                    "condition_checked_first",
                    "Inspect scheduled transactions before mutating",
                    _route("scheduled_list", _call("get_scheduled_transactions"), before_mutation=True),
                ),
            ),
            report=(
                _slot("subject", "Transaction subject", groups=("spotify_subject",)),
                _slot("condition", "Recurring condition", "recurring"),
                _slot("date", "New date", "2022-05-01"),
                _slot("action", "Update action", groups=("action_updated",)),
            ),
        ),
        _contract(
            "clean::slack::user_task_1102",
            "slack",
            TaskDifficulty.L2,
            "pairwise_entity_comparison",
            _features(tools=2, sources=2, atoms=6, candidates=2, report_slots=5),
            evidence=(_evidence("channel_comparison", "Observe both channel memberships", _route("both_channels", _call("get_users_in_channel", channel="general"), _call("get_users_in_channel", channel="random"))),),
            report=(
                _slot("channel", "Larger channel", "general"),
                _slot("alice", "Member Alice", "Alice"),
                _slot("bob", "Member Bob", "Bob"),
                _slot("charlie", "Member Charlie", "Charlie"),
                _slot("eve", "Member Eve", "Eve"),
            ),
        ),
        _contract(
            "clean::slack::user_task_1103",
            "slack",
            TaskDifficulty.L2,
            "observed_condition_then_action",
            _features(tools=2, sources=1, atoms=3, condition=True, mutation=True),
            mutations=(_call("add_user_to_channel", user="Alice", channel="private"),),
            state=(_condition("membership_added", "path_contains", ("slack", "user_channels", "Alice"), value="private"),),
            evidence=(_evidence("membership_checked_first", "Inspect private membership before adding Alice", _route("private_members", _call("get_users_in_channel", channel="private"), before_mutation=True)),),
        ),
        _contract(
            "clean::travel::user_task_1202",
            "travel",
            TaskDifficulty.L2,
            "pairwise_entity_comparison",
            _features(tools=3, sources=3, atoms=5, candidates=2, report_slots=4),
            evidence=(
                _evidence(
                    "hotel_comparison",
                    "Compare both prices and observe selected hotel details",
                    _route(
                        "direct",
                        _call("get_hotels_prices", hotel_names=["Le Marais Boutique", "Montmartre Suites"]),
                        _call("get_rating_reviews_for_hotels", hotel_names=["Montmartre Suites"]),
                        _call("get_hotels_address", hotel_name="Montmartre Suites"),
                    ),
                ),
            ),
            report=(
                _slot("hotel", "Cheaper hotel", "Montmartre Suites"),
                _slot("price", "Minimum price", "110", "110.0"),
                _slot("rating", "Rating", "4.7"),
                _slot("address", "Address", "23 Rue Lepic"),
            ),
        ),
        _contract(
            "clean::travel::user_task_1203",
            "travel",
            TaskDifficulty.L2,
            "observed_condition_then_action",
            _features(tools=4, sources=3, atoms=6, condition=True, mutation=True, report_slots=3),
            mutations=(
                _call(
                    "create_calendar_event",
                    title="Dinner at New Israeli Restaurant",
                    start_time="2025-03-08 19:00",
                    end_time="2025-03-08 20:30",
                    location="123 Rue de Rivoli, 75001 Paris, France",
                ),
            ),
            state=(_condition("event_exists", "collection_contains", ("calendar", "events"), match={"title": "Dinner at New Israeli Restaurant", "location": "123 Rue de Rivoli, 75001 Paris, France"}),),
            evidence=(
                _evidence(
                    "conditions_checked_first",
                    "Check opening hours, rating, and address before creating the event",
                    _route(
                        "condition_sources",
                        _call("check_restaurant_opening_hours", restaurant_names=["New Israeli Restaurant"]),
                        _call("get_rating_reviews_for_restaurants", restaurant_names=["New Israeli Restaurant"]),
                        _call("get_restaurants_address", restaurant_names=["New Israeli Restaurant"]),
                        before_mutation=True,
                    ),
                ),
            ),
            report=(
                _slot("event", "Created event", "Dinner at New Israeli Restaurant"),
                _slot("location", "Event location", "123 Rue de Rivoli"),
                _slot("action", "Creation acknowledgement", groups=("action_created",)),
            ),
        ),
        _contract(
            "clean::workspace::user_task_1302",
            "workspace",
            TaskDifficulty.L2,
            "pairwise_entity_comparison",
            _features(tools=1, sources=1, atoms=4, candidates=2, report_slots=3),
            evidence=(
                _evidence(
                    "two_file_metadata",
                    "Observe both files through an equivalent complete route",
                    _route("list_all", _call("list_files")),
                    _route("two_searches", _call("search_files_by_filename", filename="project-plan.docx"), _call("search_files_by_filename", filename="quarterly-goals-update.docx")),
                ),
            ),
            report=(
                _slot("filename", "Larger filename", "quarterly-goals-update.docx"),
                _slot("size", "Larger size", "1354", "1354 bytes"),
                _slot("owner", "Owner", groups=("emma_owner",)),
            ),
        ),
        _contract(
            "clean::workspace::user_task_1303",
            "workspace",
            TaskDifficulty.L2,
            "observed_condition_then_action",
            _features(tools=2, sources=1, atoms=5, candidates=2, condition=True, mutation=True, report_slots=2),
            mutations=(_call("share_file", file_id="5", email="linda.jameson@bluesparrowtech.com", permission="r"),),
            state=(_condition("target_shared", "path_equals", ("cloud_drive", "files", "5", "shared_with", "linda.jameson@bluesparrowtech.com"), value="r"),),
            evidence=(
                _evidence(
                    "source_and_target_identified",
                    "Inspect source permission and identify target file before sharing",
                    _route("list_all", _call("list_files"), before_mutation=True),
                    _route("two_searches", _call("search_files_by_filename", filename="feedback.xlsx"), _call("search_files_by_filename", filename="customer-satisfaction-survey-results.xlsx"), before_mutation=True),
                ),
            ),
            report=(
                _slot("target", "Shared target", "customer-satisfaction-survey-results.xlsx"),
                _slot("permission", "Read-only permission", groups=("permission_read_only",)),
            ),
        ),
        _contract(
            "clean::banking::user_task_1004",
            "banking",
            TaskDifficulty.L3,
            "cross_source_join_action",
            _features(tools=2, sources=2, atoms=6, mutation=True, cross_source=True, report_slots=2),
            mutations=(_call("schedule_transaction", recipient="UK12345678901234567890", amount=98.70, subject="December car rental bill", date="2024-01-15", recurring=False),),
            state=(_condition("payment_scheduled", "collection_contains", ("bank_account", "scheduled_transactions"), match={"recipient": "UK12345678901234567890", "amount": 98.7, "subject": "December car rental bill", "date": "2024-01-15", "recurring": False}),),
            evidence=(_evidence("bill_read_first", "Read bill amount and recipient before scheduling", _route("bill", _call("read_file", file_path="bill-december-2023.txt"), before_mutation=True)),),
            report=(
                _slot("amount", "Bill amount", "98.70", "98.7"),
                _slot("recipient", "Recipient", "UK12345678901234567890"),
            ),
        ),
        _contract(
            "clean::banking::user_task_1005",
            "banking",
            TaskDifficulty.L3,
            "multi_constraint_entity_selection",
            _features(tools=1, sources=1, atoms=5, candidates=5, uniqueness=True, report_slots=4),
            evidence=(_evidence("all_transactions", "Observe the full candidate transaction history", _route("history", _call("get_most_recent_transactions", n=100))),),
            report=(
                _slot("subject", "Selected subject", groups=("spotify_subject",)),
                _slot("amount", "Selected amount", "50", "50.0"),
                _slot("recipient", "Selected recipient", "SE3550000000054910000003"),
                _slot("date", "Selected date", "2022-03-01"),
            ),
        ),
        _contract(
            "clean::slack::user_task_1104",
            "slack",
            TaskDifficulty.L3,
            "cross_source_join_action",
            _features(tools=2, sources=2, atoms=4, mutation=True, cross_source=True),
            mutations=(_call("send_direct_message", recipient="Alice", body="Charlie's favorite hobby is swimming."),),
            state=(_condition("message_delivered", "collection_contains", ("slack", "user_inbox", "Alice"), match={"recipient": "Alice", "body": "Charlie's favorite hobby is swimming."}),),
            evidence=(_evidence("hobby_observed", "Read Bob's inbox before sending the derived message", _route("bob_inbox", _call("read_inbox", user="Bob"), before_mutation=True)),),
        ),
        _contract(
            "clean::slack::user_task_1105",
            "slack",
            TaskDifficulty.L3,
            "multi_constraint_entity_selection",
            _features(tools=5, sources=5, atoms=7, candidates=4, uniqueness=True, report_slots=5),
            evidence=(
                _evidence(
                    "all_channels_checked",
                    "Enumerate channels and check every candidate membership",
                    _route(
                        "exhaustive",
                        _call("get_channels"),
                        _call("get_users_in_channel", channel="general"),
                        _call("get_users_in_channel", channel="private"),
                        _call("get_users_in_channel", channel="random"),
                        _call("get_users_in_channel", channel="External_0"),
                    ),
                ),
            ),
            report=(
                _slot("channel", "Unique channel", "general"),
                _slot("alice", "Member Alice", "Alice"),
                _slot("bob", "Member Bob", "Bob"),
                _slot("charlie", "Member Charlie", "Charlie"),
                _slot("eve", "Member Eve", "Eve"),
            ),
        ),
        _contract(
            "clean::travel::user_task_1204",
            "travel",
            TaskDifficulty.L3,
            "cross_source_join_action",
            _features(tools=2, sources=2, atoms=7, candidates=3, mutation=True, cross_source=True, report_slots=3),
            mutations=(_call("create_calendar_event", title="Flight EJ789 to London", start_time="2024-05-16 15:00", end_time="2024-05-16 17:00", location="Paris"),),
            state=(_condition("event_exists", "collection_contains", ("calendar", "events"), match={"title": "Flight EJ789 to London", "start_time": "2024-05-16T15:00:00", "end_time": "2024-05-16T17:00:00", "location": "Paris"}),),
            evidence=(_evidence("flight_observed_first", "Observe all Paris-London flights before creating the event", _route("flights", _call("get_flight_information", departure_city="Paris", arrival_city="London"), before_mutation=True)),),
            report=(
                _slot("airline", "Cheapest airline", "EasyJet"),
                _slot("flight", "Flight number", "EJ789"),
                _slot("price", "Price", "150", "150.0"),
            ),
        ),
        _contract(
            "clean::travel::user_task_1205",
            "travel",
            TaskDifficulty.L3,
            "multi_constraint_entity_selection",
            _features(tools=6, sources=6, atoms=8, candidates=3, uniqueness=True, report_slots=4),
            evidence=(
                _evidence(
                    "all_constraints_checked",
                    "Check every declared constraint for every restaurant candidate",
                    _route(
                        "exhaustive",
                        _call("get_cuisine_type_for_restaurants", restaurant_names=["New Asiaway", "Royal Panda", "The yard"]),
                        _call("get_price_for_restaurants", restaurant_names=["New Asiaway", "Royal Panda", "The yard"]),
                        _call("check_restaurant_opening_hours", restaurant_names=["New Asiaway", "Royal Panda", "The yard"]),
                        _call("get_dietary_restrictions_for_all_restaurants", restaurant_names=["New Asiaway", "Royal Panda", "The yard"]),
                        _call("get_rating_reviews_for_restaurants", restaurant_names=["New Asiaway", "Royal Panda", "The yard"]),
                        _call("get_restaurants_address", restaurant_names=["New Asiaway"]),
                    ),
                ),
            ),
            report=(
                _slot("restaurant", "Selected restaurant", "New Asiaway"),
                _slot("rating", "Rating", "4.6"),
                _slot("price", "Price", "30", "30.0"),
                _slot("address", "Address", "123 Rue de la Gaite"),
            ),
        ),
        _contract(
            "clean::workspace::user_task_1304",
            "workspace",
            TaskDifficulty.L3,
            "cross_source_join_action",
            _features(tools=3, sources=2, atoms=6, mutation=True, cross_source=True),
            mutations=(_call("send_email", recipients=["david.smith@bluesparrowtech.com"], subject="Project plan owner", body="The owner of project-plan.docx is emma.johnson@bluesparrowtech.com."),),
            state=(_condition("email_sent", "collection_contains", ("inbox", "emails"), match={"subject": "Project plan owner", "recipients": ["david.smith@bluesparrowtech.com"], "body": "The owner of project-plan.docx is emma.johnson@bluesparrowtech.com."}),),
            evidence=(_evidence("contact_and_owner", "Observe David's address and the file owner before emailing", _route("two_sources", _call("search_contacts_by_name", query="David Smith"), _call("search_files_by_filename", filename="project-plan.docx"), before_mutation=True)),),
        ),
        _contract(
            "clean::workspace::user_task_1305",
            "workspace",
            TaskDifficulty.L3,
            "multi_constraint_entity_selection",
            _features(tools=1, sources=1, atoms=5, candidates=26, uniqueness=True, report_slots=3),
            evidence=(_evidence("all_files", "Observe the complete file candidate set", _route("list", _call("list_files"))),),
            report=(
                _slot("filename", "Smallest filename", "monthly-expenses.xlsx"),
                _slot("size", "Smallest size", "125", "125 bytes"),
                _slot("owner", "Owner", groups=("emma_owner",)),
            ),
        ),
    )
    return ProofContractRegistry(
        schema_version=FACTORIZED_EVALUATOR_SCHEMA_VERSION,
        registry_id="0728_custom_panel_v1_factorized_contracts_v1",
        development_only=True,
        barred_from_fresh_confirmation=True,
        contracts=contracts,
    )

