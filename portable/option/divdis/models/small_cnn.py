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
            nn.LazyConv2d(out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=5, stride=2),
            
            nn.LazyConv2d(out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=5, stride=2),
            
            #nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            #nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(),
            nn.LazyLinear(1000),
            nn.LeakyReLU(),

            nn.LazyLinear(100),
            nn.LeakyReLU(),
            
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
        
        # print(pred)
        
        return pred
