import torch.nn as nn
from portable.option.policy.models.q_function import SingleSharedBias

from pfrl.initializers import init_chainer_default
from pfrl.q_functions import DiscreteActionValueHead

class CNNQFunction(nn.Module):
    def __init__(self,
                 n_actions):
        super().__init__()
        
        self.q_func = nn.Sequential(
            nn.LazyConv2d(out_channels=16, kernel_size=3, stride=1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), 
            
            nn.LazyConv2d(out_channels=32, kernel_size=3, stride=1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Flatten(),
            init_chainer_default(nn.LazyLinear(1000)),
            init_chainer_default(nn.LazyLinear(n_actions, bias=False)),
            SingleSharedBias(),
            DiscreteActionValueHead()
        )
    
    def forward(self, x):
        return self.q_func(x)


