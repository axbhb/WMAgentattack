import torch
from wmagentattack.predicted_event_graph_dynamics import PredictedEventGraphDynamics


def model():return PredictedEventGraphDynamics(graph_size=263,candidate_size=128,hidden_size=96,dropout=0)
def test_zero_gate_preserves_observed_context():
    m=model();h=torch.randn(4,96);g=torch.randn(4,263);torch.testing.assert_close(m.condition(h,g),h,rtol=0,atol=0)
def test_predict_and_advance_shapes():
    m=model();h=torch.randn(4,96);a=torch.randn(4,128);next_h,logits=m.advance_predicted(h,a);assert next_h.shape==(4,96);assert logits.shape==(4,263)
def test_prediction_depends_on_action():
    m=model();h=torch.randn(3,96);a=torch.randn(3,128);b=a.clone();b[0]+=1;assert not torch.allclose(m.predict_graph_logits(h,a),m.predict_graph_logits(h,b))
