import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class PrintLayer(torch.nn.Module):
    # print input. For debugging
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        print(x.shape)
        
        return x

class SmallCNN(nn.Module):
    def __init__(self,
                 num_input_channels,
                 num_classes,
                 num_heads):
        super().__init__()
        
        self.model = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=num_input_channels,
                      out_channels=32,
                      kernel_size=(3,3),
                      stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=(3,3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Flatten(),
            nn.LazyLinear(1000),
            nn.ReLU(),
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
                y = F.softmax(self.model[idx](x))
            pred[:,idx,:] = y
        
        # print(pred)
        
        return pred
