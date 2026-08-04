from types import SimpleNamespace

from pydantic import BaseModel

from wmagentattack.clean_state_instrumentation import (
    build_ground_truth_goal_slots,
    candidate_tool_manifest,
    instrument_function_call,
    match_completed_goal_slots,
    normalized_argument_slots,
)


class FakeEnvironment(BaseModel):
    users: list[dict]
    audit: list[str] = []


class FakeArguments(BaseModel):
    user_id: str
    email: str


class FakeRuntime:
    def __init__(self):
        self.functions = {
            "update_user": SimpleNamespace(
                description="Update one synthetic user.",
                parameters=FakeArguments,
                dependencies={"users": object()},
            )
        }

    def run_function(self, environment, function, arguments):
        if function != "update_user":
            return "", "ToolNotFoundError: unavailable"
        for user in environment.users:
            if user["id"] == arguments["user_id"]:
                user["email"] = arguments["email"]
                environment.audit.append("updated")
                return {"updated": True}, None
        return "", "LookupError: user not found"


def test_exact_transition_records_state_delta_and_no_outcome_or_raw_output():
    environment = FakeEnvironment(
        users=[{"id": "user-1", "email": "old@example.test"}]
    )
    runtime = FakeRuntime()
    transition, output = instrument_function_call(
        runtime,
        environment,
        event_index=0,
        function="update_user",
        arguments={"user_id": "user-1", "email": "new@example.test"},
    )
    assert output == {"updated": True}
    assert transition.tool_execution_status == "success"
    assert transition.state_changed is True
    assert transition.state_before_fingerprint != transition.state_after_fingerprint
    assert transition.normalized_argument_slots == ("/email", "/user_id")
    links = {row.argument_path: row for row in transition.argument_entity_links}
    assert links["/user_id"].resolution == "unique"
    assert links["/user_id"].state_paths == ("/users/0/id",)
    assert links["/email"].resolution == "no_match"
    payload = transition.model_dump(mode="json")
    assert "utility" not in payload
    assert "security" not in payload
    assert "tool_output" not in payload
    assert transition.task_progress_delta is None
    assert transition.irreversible_effect is None


def test_error_transition_retains_error_type_without_guessing_progress():
    environment = FakeEnvironment(users=[])
    transition, _ = instrument_function_call(
        FakeRuntime(),
        environment,
        event_index=1,
        function="missing_tool",
        arguments={},
    )
    assert transition.tool_execution_status == "error"
    assert transition.tool_error_type == "ToolNotFoundError"
    assert transition.state_changed is False
    assert transition.canonical_state_delta == ()


def test_goal_slots_are_target_only_and_exact_calls_are_not_double_counted():
    calls = [
        {"function": "update_user", "args": {"user_id": "user-1", "email": "a"}},
        {"function": "update_user", "args": {"user_id": "user-1", "email": "a"}},
    ]
    slots = build_ground_truth_goal_slots(
        calls,
        {"users": [{"id": "user-1", "email": "old"}]},
    )
    completed, remaining = match_completed_goal_slots([calls[0]], slots)
    assert completed == ("goal-slot-000",)
    assert remaining == ("goal-slot-001",)
    assert all(slot.target_only for slot in slots)


def test_candidate_manifest_exposes_schema_but_not_invented_preconditions():
    manifest = candidate_tool_manifest(FakeRuntime())
    assert manifest[0]["name"] == "update_user"
    assert manifest[0]["dependencies"] == ["users"]
    assert manifest[0]["precondition_metadata_available"] is False
    assert manifest[0]["irreversibility_metadata_available"] is False
    assert normalized_argument_slots({"nested": {"id": 1}}) == ("/nested/id",)
