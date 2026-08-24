import copy

import numpy as np
import torch

from wmagentattack.support_conditioned_effect_world_model import (
    SupportConditionedEffectTransition,
    atom_target_matrix,
    compose_effect_probabilities,
    effect_slot_atoms,
    matched_count_targets,
    support_atom_target_matrix,
)


def test_effect_atoms_are_compositional():
    assert effect_slot_atoms("attribute=bank_account::balance::SINGLE_VALUED") == (
        "category::attribute", "entity::bank_account", "field::balance", "kind::SINGLE_VALUED"
    )


def test_ordinal_head_is_ordered_and_normalized():
    model = SupportConditionedEffectTransition(5, 7, 11, 13)
    hidden = torch.randn(8, 11)
    probability = model.ordinal_probabilities(hidden)
    assert torch.all(probability >= 0)
    assert torch.allclose(probability.sum(-1), torch.ones(8), atol=1e-6)
    thresholds = model.ordinal_thresholds()
    assert torch.all(thresholds[1:] > thresholds[:-1])


def test_composer_uses_atoms_and_exact_ordinal_count():
    atoms = ["category::entity", "entity::bank_account"]
    atom_probability = np.asarray([[0.81, 1.0]], dtype=np.float32)
    counts = np.asarray([[0.05, 0.10, 0.15, 0.70]], dtype=np.float32)
    output = compose_effect_probabilities(
        atom_probability, counts, ["entity=bank_account", "matched_count=3"], atoms
    )
    assert np.isclose(output[0, 0], 0.9)
    assert np.isclose(output[0, 1], 0.7)


def test_support_loader_never_reads_audit_only():
    rows = [{
        "model_target": {"effect_slot_atoms": ["category::entity"]},
        "audit_only": {"composite_effect_tokens": ["FORBIDDEN"]},
    }]
    without_audit = copy.deepcopy(rows)
    del without_audit[0]["audit_only"]
    left = support_atom_target_matrix(rows, ["category::entity"])
    right = support_atom_target_matrix(without_audit, ["category::entity"])
    assert np.array_equal(left, right)


def test_hard_targets_and_count_targets_are_exact():
    rows = [["entity=file", "matched_count=0"], ["attribute=file::name::SINGLE_VALUED", "matched_count=3"]]
    atoms = [
        "category::attribute", "category::entity", "entity::file", "field::name", "kind::SINGLE_VALUED"
    ]
    target = atom_target_matrix(rows, atoms)
    assert target.shape == (2, 5)
    assert matched_count_targets(rows).tolist() == [0, 3]
