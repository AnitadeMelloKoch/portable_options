import torch
import torch.nn as nn

from portable.procgen.models.impala import ConvSequence


class ProcgenCNN(nn.Module):
    """
    custom CNN architecture for learning on procgen
    this is designed to resemble the ImpalaCNN architecture, but to work with DoubleDQN
    """
    def __init__(self, obs_space, num_outputs):
        super().__init__()

        c, h, w = obs_space.shape
        shape = (c, h, w)
        
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        self.conv_seqs = nn.ModuleList(conv_seqs)
        self.hidden_fc = nn.Linear(in_features=shape[0] * shape[1] * shape[2],
                                  out_features=256)
        self.q_func = nn.Linear(in_features=256, out_features=num_outputs, bias=False)
        # initialize the weights
        nn.init.orthogonal_(self.q_func.weight, gain=0.01)

    def forward(self, obs):
        assert obs.ndim == 4
        x = obs / 255.0
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(x)
        x = self.hidden_fc(x)
        x = torch.relu(x)
        x = self.q_func(x)
        return x
