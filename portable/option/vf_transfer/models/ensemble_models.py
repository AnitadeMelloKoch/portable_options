import torch.nn as nn 
from pfrl.q_functions import DiscreteActionValueHead
import torch
import gin

class DQNHead(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
    
    def forward(self, x):
        return self.head(x)

@gin.configurable
class DQNEnsemble(nn.Module):
    def __init__(self, num_heads, num_actions):
        super().__init__()
        
        self.num_heads = num_heads
        self.num_actions = num_actions
        
        self.shared = nn.Sequential(
            nn.LazyConv2d(out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        self.heads = nn.ModuleList([DQNHead(num_actions) for _ in range(num_heads)])
        
    def forward(self, x):
        out = self.shared(x)
        q_values = [h(out) for h in self.heads]
        q_values = torch.stack(q_values, dim=0)
        q_mean = q_values.mean(dim=0)  # (batch_size, num_actions)
        q_std = q_values.std(dim=0)    # (batch_size, num_actions)
        return q_mean, q_std

    def forward_all_heads(self, x):
        out = self.shared(x)
        return [h(out) for h in self.heads]  # list of (batch_size, num_actions)
            
        
    