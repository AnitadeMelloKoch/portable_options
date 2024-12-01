import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

# Define the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Pretrained CLIP model name
clip_model_name = "openai/clip-vit-base-patch32"

class Clip(nn.Module):
    def __init__(self, num_classes, num_heads):
        super().__init__()

        self.embedding_class = torch.hub.load('openai/CLIP', 'clip_vit_b32', pretrained=True)

        # Custom layers for predictions
        self.model = nn.ModuleList([
            nn.Sequential(
                nn.LazyLinear(128),
                nn.LazyLinear(64),
                nn.LazyLinear(num_classes)
            ) 
            for _ in range(num_heads)])

        self.num_heads = num_heads
        self.num_classes = num_classes

    def forward(self, x):
        if x.ndim == 3:  # If missing channel dimension
            x = x.unsqueeze(1)  # Add channel dimension
        if x.shape[1] == 1:  # If grayscale, repeat to make RGB
            x = x.repeat(1, 3, 1, 1)
            
        embedding = x.mean(dim=(2, 3))
        pred = torch.zeros(x.shape[0], self.num_heads, self.num_classes).to(x.device)
        for idx in range(self.num_heads):
            y = self.model[idx](embedding)
            pred[:, idx, :] = y
            # Apply softmax to get probabilities for each class
        pred = F.softmax(pred, dim=-1)
        return pred

# # Example usage:
# if __name__ == "__main__":
#     # Number of classes and heads
#     num_classes = 10
#     num_heads = 3

#     # Create the model
#     clip_model = Clip(num_classes=num_classes, num_heads=num_heads)

#     # Dummy image batch (e.g., PIL images or similar)
#     from PIL import Image
#     dummy_images = [Image.new("RGB", (224, 224), color="white") for _ in range(4)]

#     # Forward pass
#     outputs = clip_model(dummy_images)
#     print(outputs.shape)  # Expected shape: (batch_size, num_heads, num_classes)