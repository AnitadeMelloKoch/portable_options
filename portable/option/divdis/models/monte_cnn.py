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

class MonteCNN(nn.Module):
    def __init__(self,
                 num_classes,
                 num_heads):
        super().__init__()

        self.model = nn.ModuleList([nn.Sequential(
            nn.LazyConv2d(out_channels=32, kernel_size=8, stride=4, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.LazyConv2d(out_channels=64, kernel_size=4, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.LazyConv2d(out_channels=64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Flatten(),           
            nn.LazyLinear(512),
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
