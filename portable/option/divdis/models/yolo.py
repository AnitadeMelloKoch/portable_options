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
        # self.full_model = nn.ModuleList([nn.Sequential([self.embedding_class, self.model]) for _ in range(num_heads))
        
        # Full model includes embedding and classification head
        self.full_model = nn.ModuleList([
            nn.Sequential(self.embedding_class, classification_head)
            for classification_head in self.model
        ])

    def forward(self, x):
        print("x shape:", x.shape)
        # Access backbone and neck layers (not including the final detection head)
        
        x = self.embedding_class.model.model[0](x)  # Backbone
        x = self.embedding_class.model.model[1](x)  # Neck

        print("before embedding shape:", x.shape)
        # Global average pooling over spatial dimensions (height, width)
        embedding = x.mean(dim=(2, 3))  # Global average pooling to [batch_size, channels]
        embedding = embedding.to(x.device)
        # Define a linear layer to transform from 64 to 85 dimensions
        linear_layer = nn.Linear(64, 85).to(x.device)

        # Assuming embedding is of shape [batch_size, 64]
        embedding = linear_layer(embedding)
        print("embedding shape:", embedding.shape)


        # Prediction logic: apply custom model layers on the embedding
        pred = torch.zeros(x.shape[0], self.num_heads, self.num_classes).to(x.device)
        for idx in range(self.num_heads):
            y = self.model[idx](embedding)
            print("y shape:", y.shape)
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
