import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


class Clip(nn.Module):
    def __init__(self, num_classes, num_heads):
        super().__init__()

        # Load the pretrained CLIP model
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda" if torch.cuda.is_available() else "cpu")

        # Define custom model layers for each head
        self.model = nn.ModuleList([
            nn.Sequential(
                nn.LazyLinear(512),
                nn.LazyLinear(256),  # Adjust dimensions as needed
                nn.LazyLinear(num_classes)  # Final output layer for classification
            ) for _ in range(num_heads)
        ])

        self.num_heads = num_heads
        self.num_classes = num_classes

    def forward(self, images):
        # Ensure images have the correct shape [batch_size, 3, height, width]
        if images.dim() == 3:  # If missing batch dimension
            images = images.unsqueeze(0)  # Add batch dimension
            images = images.repeat(1, 3, 1, 1)  # Ensure RGB channels

        if images.dim() == 4:  # Ensure only 3 channels are used
            images = images[:, -3:, :, :]

        # Move images to the device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        images = images.to(device)

        # Extract image features using CLIP (mocked here)
        # Replace this with `get_image_features` when using the CLIP model
        embeddings = images.mean(dim=(2, 3))

        # Apply custom model layers for each head
        batch_size = images.size(0)
        predictions = torch.zeros(batch_size, self.num_heads, self.num_classes).to(device)

        for idx in range(self.num_heads):
            y = self.model[idx](embeddings)
            predictions[:, idx, :] = y

        # Apply softmax with temperature scaling to adjust prediction confidence
        temperature = 0.5  # Temperature < 1 sharpens predictions
        predictions = F.softmax(predictions / temperature, dim=-1)

        return predictions