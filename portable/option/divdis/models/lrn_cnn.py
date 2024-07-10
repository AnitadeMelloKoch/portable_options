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
        print(x)
        return x

class LRN(nn.Module):
    def __init__(self, local_size=3, alpha=1.0, beta=1.0, ACROSS_CHANNELS=True, k=1e-6):
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

class LRN_CNN(nn.Module):
    def __init__(self,
                 num_classes,
                 num_heads):
        super().__init__()

        self.model = nn.ModuleList([nn.Sequential(
            # batch_size x 4 x 84 x 84
            nn.LazyConv2d(out_channels=32, kernel_size=5, stride=2, padding=0, bias=False),
            # batch_size x 32 x 42 z 42
            nn.ReLU(),
            PrintLayer(),
            LRN(),
            #PrintLayer(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.LazyConv2d(out_channels=64, kernel_size=3, stride=2, padding=0, bias=False),
            nn.ReLU(),
            LRN(),
            nn.MaxPool2d(kernel_size=4, stride=2), # maybe try global avg pool in future
            
            nn.Flatten(),           
            nn.LazyLinear(750),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.LazyLinear(100),
            nn.ReLU(),
            nn.LazyLinear(num_classes)
            
            ) for _ in range(num_heads)])
        
        self.num_heads = num_heads
        self.num_classes = num_classes
    
    def forward(self, x, logits=False):
        pred = torch.zeros(x.shape[0], self.num_heads, self.num_classes).to(x.device)
        
        for idx in range(self.num_heads):
            if logits:
                y = self.model[idx](x)
            else:
                y = F.softmax(self.model[idx](x), dim=-1)
            pred[:,idx,:] = y
                
        return pred