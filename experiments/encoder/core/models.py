import torch 
import torch.nn as nn 

MODEL_TYPE = [
    "one-layer"
]

class SimpleClassifier(nn.Module):
    def __init__(self,
                 input_size):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 2),
            nn.Softmax(-1)
        )
    
    def forward(self, x):
        return self.network(x)