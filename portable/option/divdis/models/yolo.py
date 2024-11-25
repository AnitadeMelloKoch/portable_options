import torch
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np


class YOLOEnsemble(nn.Module):
    def __init__(self,
                 num_classes,
                 num_heads):
        super().__init__()

        #self.embedding_class = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')
        self.embedding_class = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).model
        
        self.model = nn.ModuleList([
            nn.Sequential(
                nn.LazyLinear(1000),
                nn.LazyLinear(700),
                nn.LazyLinear(num_classes)
                # add at least 3 more linear layers
            )
         for _ in range(num_heads)])
    
        self.num_heads = num_heads
        self.num_classes = num_classes

    def forward(self, x):
        print("Forward x:", x)
        # Ensure input is [batch_size, 3, height, width]
        if x.ndim == 3:  # If missing channel dimension
            x = x.unsqueeze(1)  # Add channel dimension
        if x.shape[1] == 1:  # If grayscale, repeat to make RGB
            x = x.repeat(1, 3, 1, 1)

        # Access backbone and neck layers (not including the final detection head)
        x = self.embedding_class.model.model[0](x)  # Backbone
        x = self.embedding_class.model.model[1](x)  # Neck

        # Global average pooling over spatial dimensions (height, width)
        embedding = x.mean(dim=(2, 3))  # Global average pooling to [batch_size, channels]


        # Prediction logic: apply custom model layers on the embedding
        pred = torch.zeros(x.shape[0], self.num_heads, self.num_classes).to(x.device)
        for idx in range(self.num_heads):
            y = self.model[idx](embedding)
            pred[:, idx, :] = y
            # Apply softmax to get probabilities for each class
        pred = F.softmax(pred, dim=-1)
        return pred
        # pred = torch.zeros(x.shape[0], self.num_heads, self.num_classes).to(x.device)
        # # embedding = torch.tensor(self.embedding_class(x))
        # with torch.no_grad():  # Avoid gradients for the embedding
        #     # embedding = self.embedding_class.model[:-1](x)  # Adjust as needed for the last usable layer
        #     features = self.embedding_class(x)  # This gives detection output
        #     embedding = features[0].mean(dim=1)
        #     print(embedding.shape)
        #     # print("Post-pooling embedding shape:", embedding.shape)  # Check the shape


        # for idx in range(self.num_heads):
        #     y = self.model[idx](embedding)
        #     pred[:,idx,:] = y

