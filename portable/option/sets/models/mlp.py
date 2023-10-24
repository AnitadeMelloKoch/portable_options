import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, input_size=64, class_num=2):
        super(MLP, self).__init__()

        self.input_num = input_size
        self.out_size = class_num

        self.linear1 = nn.Linear(self.input_num, 32)
        self.linear2 = nn.Linear(32, 16)
        self.linear3 = nn.Linear(16, self.out_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        
        return F.softmax(self.linear3(x))
        # return F.sigmoid(self.linear2(x))