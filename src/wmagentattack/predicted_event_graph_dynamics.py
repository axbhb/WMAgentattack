"""Modular action dynamics with a learned future event-graph predictor."""

from __future__ import annotations
import torch
from torch import Tensor,nn
from .structured_residual_dynamics import StructuredResidualDynamics


class PredictedEventGraphDynamics(nn.Module):
    def __init__(self,*,graph_size:int,candidate_size:int,hidden_size:int,dropout:float):
        super().__init__();self.graph_size=graph_size
        self.base=StructuredResidualDynamics(candidate_size=candidate_size,hidden_size=hidden_size,dropout=dropout)
        self.graph_encoder=nn.Sequential(nn.Linear(graph_size,hidden_size),nn.LayerNorm(hidden_size),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden_size,hidden_size),nn.LayerNorm(hidden_size))
        self.graph_predictor=nn.Sequential(nn.Linear(hidden_size*2,hidden_size),nn.LayerNorm(hidden_size),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden_size,graph_size))
        self.graph_gate=nn.Parameter(torch.zeros(()))
    def condition(self,hidden:Tensor,graph:Tensor)->Tensor:
        return hidden+torch.tanh(self.graph_gate)*self.graph_encoder(graph)
    def predict_graph_logits(self,hidden:Tensor,action_inputs:Tensor)->Tensor:
        action=self.base.action_encoder(action_inputs)
        return self.graph_predictor(torch.cat((hidden,action),dim=-1))
    def advance_predicted(self,hidden:Tensor,action_inputs:Tensor)->tuple[Tensor,Tensor]:
        graph_logits=self.predict_graph_logits(hidden,action_inputs)
        advanced=self.base.advance(hidden,action_inputs)
        return self.condition(advanced,torch.sigmoid(graph_logits)),graph_logits
    def one_step_delta_logits(self,hidden:Tensor,candidates:Tensor)->Tensor:return self.base.one_step_delta_logits(hidden,candidates)
    def rollout_logits(self,hidden:Tensor,candidates:Tensor)->Tensor:return self.base.rollout_logits(hidden,candidates)
    def projected_context(self,hidden:Tensor)->Tensor:return self.base.projected_context(hidden)


def trainable_parameter_count(model:nn.Module)->int:return sum(p.numel() for p in model.parameters() if p.requires_grad)
