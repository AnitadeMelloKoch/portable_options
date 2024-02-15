import torch 
import torch.nn as nn 

from pfrl.initializers import init_chainer_default
from pfrl.q_functions import DiscreteActionValueHead
from portable.option.policy.models.q_function import SingleSharedBias
import gin

@gin.configurable
class LargeLinearQFunction(nn.Module):
    def __init__(self,
                 in_features,
                 n_actions,
                 hidden_size=64):
        super().__init__()
        self.q_func = nn.Sequential(
            init_chainer_default(nn.Linear(in_features, hidden_size)),
            init_chainer_default(nn.Linear(hidden_size, hidden_size)),
            init_chainer_default(nn.Linear(hidden_size, n_actions, bias=False)),
            SingleSharedBias(),
            DiscreteActionValueHead()
        )
    
    def forward(self, x):
        return self.q_func(x)

@gin.configurable
class OptionLinearQFunction(nn.Module):
    def __init__(self,
                 in_features,
                 action_vector_size,
                 num_options,
                 hidden_size=64):
        super().__init__()
        self.q_func = nn.Sequential(
            init_chainer_default(nn.Linear(in_features+action_vector_size, hidden_size)),
            init_chainer_default(nn.Linear(hidden_size, hidden_size)),
            init_chainer_default(nn.Linear(hidden_size, num_options, bias=False)),
            SingleSharedBias(),
            DiscreteActionValueHead()
        )
    
    def forward(self, action_output, x):
        concat_input = torch.cat((x, action_output), dim=-1)
        return self.q_func(concat_input)
        
        

