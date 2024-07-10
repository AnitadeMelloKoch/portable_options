import torch
import os
import torch.nn as nn
import numpy as np
import torch.nn.functional as F 

class PrintLayer(torch.nn.Module):
    # print input. For debugging
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        # print(x.shape)
        return x

class LRN(nn.Module):
    def __init__(self, local_size=3, alpha=1.0, beta=1.0, ACROSS_CHANNELS=True, k=0):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.local_size = local_size


    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        x = x.div(div).div(self.local_size) # maybe delete this last part
        return x

x = np.array([[[[1, 2, 3], [4, 3, 6], [7, 8, 9]],
              [[1, 2, 1], [2, 3, 2], [3, 4, 3]],
              [[2, 1, 2], [3, 2, 3], [4, 3, 4]],
              [[4, 2, 1], [5, 2, 1], [2, 2, 4]]]])
x = x.astype(np.float32)
x = torch.from_numpy(x)
model = LRN()
print("X shape:", x.shape)
print(model.forward(x))
