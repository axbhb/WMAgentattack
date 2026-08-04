from wmagentattack.decision_state import VictimActionEvent
from wmagentattack.exact_simulator import (
    ExactSandboxSimulator,
    ToolExecution,
    canonical_state_delta,
)


def test_exact_simulator_uses_executor_and_checker_instead_of_learning_transition():
    initial = {"messages": [], "balance": 10}

    def execute(state, action):
        assert action.tool_name == "send_message"
        state["messages"].append(action.arguments["text"])
        return ToolExecution(next_state=state, output={"ok": True})

    simulator = ExactSandboxSimulator(
        execute,
        utility_checker=lambda state: len(state["messages"]) == 1,
        security_checker=lambda state: "unsafe" in state["messages"],
    )
    action = VictimActionEvent(
        event_index=0,
        tool_name="send_message",
        skill_name="message_external",
        arguments={"text": "safe"},
    )
    transition = simulator.transition(initial, action)
    assert initial == {"messages": [], "balance": 10}
    assert transition.state_after["messages"] == ["safe"]
    assert transition.utility is True
    assert transition.security is False
    assert transition.tool_error is None
    assert transition.state_delta == (
        {
            "op": "replace",
            "path": "/messages",
            "old": [],
            "value": ["safe"],
        },
    )


def test_state_delta_has_deterministic_add_remove_replace_operations():
    delta = canonical_state_delta(
        {"a": 1, "nested": {"old": 2}},
        {"a": 3, "nested": {"new": 4}},
    )
    assert [item["op"] for item in delta] == ["replace", "remove", "add"]

