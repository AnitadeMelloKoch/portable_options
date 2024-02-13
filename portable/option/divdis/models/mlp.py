import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from portable.option.divdis.divdis import to_probs

class OneHeadMLP(nn.Module):
    def __init__(self,
                 input_dim,
                 num_classes,
                 num_heads):
        super().__init__()
        
        self.model = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, input_dim//2),
                nn.ReLU(),
                nn.BatchNorm1d(input_dim//2),
                nn.Linear(input_dim//2, num_classes)
            )
        for _ in range(num_heads)] )
        
        self.num_heads = num_heads
        self.num_classes = num_classes
        
    def forward(self, x, logits=False):
        pred = torch.zeros(x.shape[0], self.num_heads, self.num_classes).to(x.device)
        
        for idx in range(self.num_heads):
            if logits:
                y = self.model[idx](x)
            else:
                y = F.softmax(self.model[idx](x), dim=1)
            pred[:,idx,:] = y
        
        return pred

class MultiHeadMLP(nn.Module):
    def __init__(self,
                 input_dim,
                 num_classes,
                 num_heads):
        super().__init__()
        
        self.num_heads = num_heads
        self.num_classes = num_classes
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, num_classes*num_heads),
            nn.ReLU(),
            nn.BatchNorm1d(num_classes*num_heads),
            nn.Linear(num_classes*num_heads, num_classes*num_heads),
            nn.ReLU(),
            nn.BatchNorm1d(num_classes*num_heads),
            nn.Linear(num_classes*num_heads, num_classes*num_heads)
        )
    
    def forward(self, x, logits=False):
        if logits is False:
            return to_probs(self.model(x), self.num_heads)
        else:
            return self.model(x)

