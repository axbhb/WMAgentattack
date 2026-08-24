from wmagentattack.explicit_support_panel import effect_slot_atoms


def test_attribute_atoms_are_factorized():
    assert effect_slot_atoms("attribute=webpage::content::SINGLE_VALUED") == [
        "category::attribute",
        "entity::webpage",
        "field::content",
        "kind::SINGLE_VALUED",
    ]


def test_entity_atom_keeps_entity_identity():
    assert effect_slot_atoms("entity=bank_account") == [
        "category::entity",
        "entity::bank_account",
    ]


def test_count_atom_is_ordinal_auditable():
    assert effect_slot_atoms("matched_count=3") == [
        "category::matched_count",
        "kind::count",
        "value::3",
    ]
