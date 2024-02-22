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
            # nn.Conv2d(in_channels=num_input_channels,
            #           out_channels=96,
            #           kernel_size=(11,11),
            #           stride=4),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(3,3),
            #              stride=(2,2)),
            nn.Conv2d(in_channels=3,
                      out_channels=96,
                      kernel_size=(5,5),
                      stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3),
                         stride=(2,2)),
            nn.Conv2d(in_channels=96,
                      out_channels=256,
                      kernel_size=(3,3),
                      stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,
                      out_channels=96,
                      kernel_size=(3,3),
                      stride=1),
            nn.ReLU(),
            # nn.Conv2d(in_channels=384,
            #           out_channels=256,
            #           kernel_size=(3,3),
            #           stride=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(3,3),
            #              stride=2),
            nn.Flatten(),
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.LazyLinear(1000),
            nn.ReLU(),
            nn.LazyLinear(2)
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
