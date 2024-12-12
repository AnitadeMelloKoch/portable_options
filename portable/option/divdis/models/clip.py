import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


class Clip(nn.Module):
    def __init__(self, num_classes, num_heads):
        super().__init__()

        # Load the pretrained CLIP model (image backbone)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").vision_model
        
        # Custom classification heads
        self.model = nn.ModuleList([
            nn.Sequential(
                nn.LazyLinear(512),
                nn.ReLU(),
                nn.LazyLinear(256),
                nn.ReLU(),
                nn.LazyLinear(128),
                nn.ReLU(),
                nn.LazyLinear(num_classes)
            )
            for _ in range(num_heads)
        ])

        self.num_heads = num_heads
        self.num_classes = num_classes

    def forward(self, x):
        # Forward pass through the CLIP vision backbone
        x = self.clip_model.embeddings(x)  # Embedding layer
        x = self.clip_model.pre_layrnorm(x)  # Pre-layer normalization
        
        # Apply global average pooling over spatial dimensions (height, width)
        embedding = x.mean(dim=(1, 2))  # Pool to shape [batch_size, channels]

        # Initialize predictions tensor
        batch_size = embedding.size(0)
        pred = torch.zeros(batch_size, self.num_heads, self.num_classes, device=embedding.device)

        # Apply each classification head
        for idx in range(self.num_heads):
            pred[:, idx, :] = self.model[idx](embedding)

        # Apply softmax over the class dimension
        pred = F.softmax(pred, dim=-1)
        return pred