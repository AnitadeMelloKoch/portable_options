import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import CLIPModel

class PrintLayer(torch.nn.Module):
    # Print input for debugging
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        print(x.shape)
        return x

class GlobalAveragePooling2D(nn.Module):
    # Custom global average pooling layer
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean(dim=(2, 3))  # Average over height and width

class CLIPEensemble(nn.Module):
    def __init__(self, num_classes, num_heads):
        super().__init__()
        
        # Load pre-trained CLIP model
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        # Process input through CLIP vision encoder to extract image embeddings
        with torch.no_grad():  # Avoid gradients for pre-trained CLIP
            self.clip_model.vision_model.requires_grad_(False)
        
        # Classification heads
        self.model = nn.ModuleList([
            nn.Sequential(
                nn.LazyLinear(1000),
                nn.ReLU(),
                nn.LazyLinear(700),
                nn.ReLU(),
                nn.LazyLinear(num_classes)
            )
            for _ in range(num_heads)
        ])

        self.num_heads = num_heads
        self.num_classes = num_classes

        # Full model combines embedding and classification heads
        self.full_model = nn.ModuleList([
            nn.Sequential(
                PrintLayer(),
                nn.Sequential(
                    PrintLayer(),
                    nn.LazyLinear(self.clip_model.vision_model.config.hidden_size),
                    nn.ReLU(),
                    PrintLayer()
                ),
                GlobalAveragePooling2D(),  # Custom GAP layer
                PrintLayer(),
                classification_head,
                PrintLayer()
            )
            for classification_head in self.model
        ])

    def forward(self, x):
        print("Input shape:", x.shape)

        # Process input through CLIP vision encoder to extract image embeddings
        with torch.no_grad():
            embeddings = self.clip_model.vision_model(x).last_hidden_state
            embeddings = embeddings.mean(dim=1)  # Mean pooling over tokens

        print("Embeddings shape:", embeddings.shape)

        # Predictions for each head
        pred = torch.zeros(embeddings.shape[0], self.num_heads, self.num_classes).to(embeddings.device)
        for idx in range(self.num_heads):
            y = self.full_model[idx](embeddings)
            print(f"Head {idx} output shape:", y.shape)
            pred[:, idx, :] = y

        # Apply softmax to get probabilities
        pred = F.softmax(pred, dim=-1)
        return pred
