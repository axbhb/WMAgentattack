import torch

from wmagentattack.action_event_graph_dynamics import ActionEventGraphDynamics


def model():
    return ActionEventGraphDynamics(
        graph_size=263, candidate_size=128, hidden_size=96, dropout=0.0
    )


def test_zero_gate_is_exact_context_noop():
    instance=model();hidden=torch.randn(5,96);graph=torch.randn(5,263)
    conditioned=instance.condition(hidden,graph)
    torch.testing.assert_close(conditioned,hidden,rtol=0,atol=0)


def test_graph_changes_hidden_after_gate_opens():
    instance=model();instance.graph_gate.data.fill_(0.25)
    hidden=torch.randn(5,96);left=torch.zeros(5,263);right=torch.ones(5,263)
    assert not torch.allclose(instance.condition(hidden,left),instance.condition(hidden,right))


def test_advance_preserves_batch_shape():
    instance=model();hidden=torch.randn(7,96);action=torch.randn(7,128);graph=torch.randn(7,263)
    assert instance.advance(hidden,action,graph).shape==(7,96)
