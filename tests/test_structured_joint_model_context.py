import torch
from wmagentattack.joint_outcome_auxiliary import StructuredJointOutcomeModel
def test_context_refactor_preserves_forward_score():
 m=StructuredJointOutcomeModel(state_size=10,candidate_size=8,hidden_size=16,dropout=0.0);m.eval();s=torch.randn(3,10);a=torch.randn(3,8);c=torch.randn(5,8);out=m(s,a,c);ctx=m.encode_context(s,a);assert torch.allclose(out[0],m.score_candidates(ctx,c))
