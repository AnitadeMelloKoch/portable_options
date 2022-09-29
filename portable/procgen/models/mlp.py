import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, input_size=64, hidden_size=64, class_num=2):
        super(MLP, self).__init__()

        self.input_num = input_size
        self.out_size = class_num

        self.linear1 = nn.Linear(self.input_num, hidden_size)
        self.linear2 = nn.Linear(hidden_size, self.out_size)

    def forward(self, x):

        x = F.relu(self.linear1(x))
        x = F.softmax(self.linear2(x), dim=1)
        
        return x


class PPOMLP(nn.Module):
    """"
    instead of using conv layers and residual connections (Impala) for PPO value network, we use 
    just linear layers
    """
    def __init__(self, hidden_size=256, output_size=15):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hidden = nn.LazyLinear(self.hidden_size)
        self.logits = nn.Linear(self.hidden_size, self.output_size)
        self.value = nn.Linear(self.hidden_size, 1)
        # initialize weights and bias
        nn.init.orthogonal_(self.logits.weight, gain=0.01)
        nn.init.zeros_(self.logits.bias)
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(x)
        x = self.hidden(x)
        x = torch.relu(x)
        logits = self.logits(x)
        dist = torch.distributions.Categorical(logits=logits)
        value = self.value(x)
        return dist, value
