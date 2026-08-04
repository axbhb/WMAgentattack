import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_headwise_remaining_protocol_freezes_budget_and_selection_hash():
    protocol = json.loads(
        (ROOT / "configs" / "0713_headwise_remaining_confirmation_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["research_budget"]["fresh_outcome_count"] == 140
    assert protocol["research_budget"]["no_early_stopping_from_interim_outcomes"]
    assert protocol["selection"]["sha256"] == (
        "9a986771d1bf1aedcd447457a175e56dfa0aa4cfac0fe405e463b0bad0cfc728"
    )
    assert protocol["protocol_revision"] == 2
    assert protocol["supersedes"]["fresh_outcomes_generated_before_revision"] == 0
    assert protocol["fixed_method"]["rank_probability_decoupled"]
