from __future__ import annotations

import numpy as np

from wmagentattack.shared_action_ontology import (
    action_ontology,
    ontology_candidate_vector,
)


def _descriptor(name: str, description: str, *, kind: str = "tool") -> dict:
    return {
        "source": "test",
        "kind": kind,
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_semantically_equivalent_email_actions_share_bridge_key() -> None:
    skill = _descriptor("email_external", "send_email: Sends an email")
    tool = _descriptor("GmailSendEmail", "Send an email to one or more recipients")
    assert action_ontology(skill)["bridge_key"] == action_ontology(tool)["bridge_key"]
    assert action_ontology(skill)["operation"] == "communicate"


def test_text_and_finish_are_terminal_stop_actions() -> None:
    text = _descriptor("TEXT", "Return text", kind="text_or_stop")
    finish = _descriptor("finish", "Return a final response", kind="text_or_stop")
    assert action_ontology(text)["operation"] == "stop"
    assert action_ontology(finish)["operation"] == "stop"
    assert action_ontology(text)["effect"] == "terminal"


def test_local_residual_preserves_distinct_tools_with_shared_ontology() -> None:
    left = _descriptor("get_hotel_price", "Get hotel information")
    right = _descriptor("get_hotel_rating", "Get hotel information")
    assert action_ontology(left)["bridge_key"] == action_ontology(right)["bridge_key"]
    left_vector = ontology_candidate_vector(
        left, mode="ontology_local_residual", hash_dimension=128
    )
    right_vector = ontology_candidate_vector(
        right, mode="ontology_local_residual", hash_dimension=128
    )
    assert not np.array_equal(left_vector, right_vector)
    assert np.isclose(np.linalg.norm(left_vector), 1.0)
