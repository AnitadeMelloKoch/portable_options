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
