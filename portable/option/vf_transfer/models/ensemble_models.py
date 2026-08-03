import torch.nn as nn
from pfrl.q_functions import DiscreteActionValueHead
import torch
import gin

# CNN output size for 84x84 input:
# Conv(in, 32, 8, 4) -> 20x20, Conv(32, 64, 4, 2) -> 9x9, Conv(64, 64, 3, 1) -> 7x7
# Flatten -> 64*7*7 = 3136
CNN_OUTPUT_SIZE = 3136

class DQNHead(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(CNN_OUTPUT_SIZE, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        return self.head(x)

@gin.configurable
class DQNEnsemble(nn.Module):
    def __init__(self, num_heads, num_actions, in_channels=1):
        super().__init__()

        self.num_heads = num_heads
        self.num_actions = num_actions

        self.shared = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4),
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


def _make_dqn_net(num_actions, in_channels=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(CNN_OUTPUT_SIZE, 512),
        nn.ReLU(),
        nn.Linear(512, num_actions),
    )


class DQNHeadFull(nn.Module):
    """RPF head with its own CNN encoder + fixed random prior CNN."""
    def __init__(self, num_actions, in_channels=1, beta=1.0):
        super().__init__()
        self.beta = beta
        self.head = _make_dqn_net(num_actions, in_channels)
        self.prior = _make_dqn_net(num_actions, in_channels)
        self.prior.eval()

    def train(self, mode=True):
        super().train(mode)
        self.prior.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            prior_out = self.prior(x)
        return self.head(x) + self.beta * prior_out.detach()


@gin.configurable
class DQNEnsembleFull(nn.Module):
    """Ensemble of DQNHeadFull — each head has its own CNN encoder and prior."""
    def __init__(self, num_heads, num_actions, in_channels=1, beta=1.0):
        super().__init__()
        self.num_heads = num_heads
        self.num_actions = num_actions
        self.heads = nn.ModuleList([
            DQNHeadFull(num_actions, in_channels, beta) for _ in range(num_heads)
        ])

    def forward(self, x):
        q_values = torch.stack([head(x) for head in self.heads], dim=0)
        return q_values.mean(dim=0), q_values.std(dim=0)

    def forward_all_heads(self, x):
        return [head(x) for head in self.heads]

    def uncertainty(self, x):
        q_values = torch.stack([head(x) for head in self.heads], dim=0)
        return q_values.std(dim=0)
