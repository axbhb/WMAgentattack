from wmagentattack.hard_label_confirmation import hard_effect_tokens, tool_family


def test_hard_effect_tokens_remove_only_action_implied_source() -> None:
    tokens = [
        "source=get_file_by_id",
        "entity=cloud_file",
        "attribute=cloud_file::content::SINGLE_VALUED",
        "execution=success",
    ]
    assert hard_effect_tokens(tokens) == [
        "attribute=cloud_file::content::SINGLE_VALUED",
        "entity=cloud_file",
        "execution=success",
    ]


def test_tool_family_contract() -> None:
    assert tool_family("suite::get_balance") == "query_read"
    assert tool_family("suite::send_email") == "create_send_reserve"
    assert tool_family("suite::update_password") == "mutation"
