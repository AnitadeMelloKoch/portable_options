import numpy as np
from torch import nn
import logging
import pfrl
import torch


class VFHead(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
        self.head = nn.Sequential(
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.prior = nn.Sequential(
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        with torch.no_grad():
            prior_out = self.prior(x)
        return self.head(x) + self.beta * prior_out.detach()

class VFEnsemble(nn.Module):
    def __init__(self,
                 num_heads,
                 use_encoder=True,
                 beta=1.0):
        super().__init__()
        self.use_encoder = use_encoder
        if use_encoder:
            self.encoder = nn.Sequential(
                nn.LazyConv2d(out_channels=32, kernel_size=5, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(32),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=4, stride=2),

                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=3, stride=1),

                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(64),
                nn.GELU(),

                nn.Flatten(),
                nn.LazyLinear(1024),
                nn.GELU(),
                nn.Linear(1024, 512),
            )
        self.heads = nn.ModuleList([
            VFHead(beta=beta) for _ in range(num_heads)
        ])

    def _encode(self, x):
        return self.encoder(x) if self.use_encoder else x

    def forward(self, x):
        x = self._encode(x)
        outputs = [head(x) for head in self.heads]
        outputs = torch.stack(outputs, dim=0)
        return outputs.mean(dim=0)

    def forward_all_heads(self, x):
        x = self._encode(x)
        return [head(x) for head in self.heads]

    def uncertainty(self, x):
        x = self._encode(x)
        outputs = [head(x) for head in self.heads]
        outputs = torch.stack(outputs, dim=0)
        return outputs.std(dim=0)


def _make_cnn_mlp():
    """Full CNN encoder + value MLP for a self-contained VFHeadFull."""
    return nn.Sequential(
        nn.LazyConv2d(out_channels=32, kernel_size=5, stride=2, padding=0, bias=False),
        nn.BatchNorm2d(32),
        nn.GELU(),
        nn.MaxPool2d(kernel_size=4, stride=2),

        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0, bias=False),
        nn.BatchNorm2d(64),
        nn.GELU(),
        nn.MaxPool2d(kernel_size=3, stride=1),

        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(64),
        nn.GELU(),

        nn.Flatten(),
        nn.LazyLinear(1024),
        nn.GELU(),
        nn.Linear(1024, 512),

        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )


class VFHeadFull(nn.Module):
    """
    Self-contained RPF head: each head owns its full CNN encoder so the VF loss
    trains it directly, with no dependence on a shared encoder.

    The prior is kept permanently in eval mode so its BatchNorm running stats
    never drift, making it a truly fixed random function of the raw input.
    """
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
        self.head = _make_cnn_mlp()
        self.prior = _make_cnn_mlp()
        self.prior.eval()

    def train(self, mode=True):
        super().train(mode)
        self.prior.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            prior_out = self.prior(x)
        return self.head(x) + self.beta * prior_out.detach()


class VFEnsembleFull(nn.Module):
    """Ensemble of VFHeadFull — each head has its own CNN encoder."""
    def __init__(self, num_heads, beta=1.0):
        super().__init__()
        self.heads = nn.ModuleList([VFHeadFull(beta=beta) for _ in range(num_heads)])

    def forward(self, x):
        outputs = torch.stack([head(x) for head in self.heads], dim=0)
        return outputs.mean(dim=0)

    def forward_all_heads(self, x):
        return [head(x) for head in self.heads]

    def uncertainty(self, x):
        outputs = torch.stack([head(x) for head in self.heads], dim=0)
        return outputs.std(dim=0)


class UncertainPPOModel(nn.Module):
    def __init__(self,
                 num_actions,
                 vf_heads,
                 prior_scale=1.0):
        super().__init__()

        # Policy-only encoder — updated solely by policy gradient.
        self.policy_embed = nn.Sequential(
            nn.LazyConv2d(out_channels=32, kernel_size=5, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=4, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=1),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Flatten(),
            nn.LazyLinear(1024),
            nn.GELU(),
            nn.Linear(1024, 512),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, num_actions),
            pfrl.policies.SoftmaxCategoricalHead()
        )

        # Each VF head owns its own CNN encoder — VF loss trains each head end-to-end
        # without disturbing the policy encoder or the other heads.
        self.vf_ensemble = VFEnsembleFull(vf_heads, beta=prior_scale)

    def forward(self, x):
        embed = self.policy_embed(x)
        distribs = self.policy_head(embed)
        head_vs = self.vf_ensemble.forward_all_heads(x)
        self._last_head_vs = head_vs
        return distribs, torch.stack(head_vs).mean(dim=0)

    def vf_uncertainty(self, x):
        return self.vf_ensemble.uncertainty(x)
    
        
    