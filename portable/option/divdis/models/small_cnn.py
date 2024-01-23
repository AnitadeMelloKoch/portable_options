import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class SmallCNN(nn.Module):
    def __init__(self,
                 num_input_channels,
                 num_classes,
                 num_heads):
        super().__init__()
        
        self.model = nn.ModuleList([nn.Sequential(
            nn.Conv2d(num_input_channels, 6, 5),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,16,5),
            nn.Flatten(),
            nn.Linear(16*5*5,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,num_classes)
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
        
        return pred
