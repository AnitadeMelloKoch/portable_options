import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel
from torchvision import transforms
from PIL import Image


class Clip(nn.Module):
    def __init__(self, num_classes, num_heads):
        super().__init__()

        # Load the pretrained CLIP model
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda" if torch.cuda.is_available() else "cpu")

        # Define custom model layers for each head
        self.model = nn.ModuleList([
            nn.Sequential(
                nn.LazyLinear(512),
                nn.LazyLinear(256),  # You can adjust the dimensions here as needed
                nn.LazyLinear(128),
                nn.LazyLinear(64),
                nn.LazyLinear(num_classes)  # Final output layer for classification
            ) for _ in range(num_heads)
        ])
        
        self.num_heads = num_heads
        self.num_classes = num_classes

    def forward(self, images):
        # Ensure images have the correct shape [batch_size, 3, height, width]
        if images.dim() == 3:  # If missing batch dimension
            images = images.unsqueeze(1)  # Add batch dimension
            # if images.shape[1] == 1:  # If grayscale, repeat channels to make RGB
            images = images.repeat(1, 3, 1, 1)
        if images.dim() == 4:
            images = images[:, -3:, :, :] # Ensure only 3 channels are used

        # Move the images to the same device as the model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        images = images.to(device)

        # # Extract image features using CLIP (image embeddings)
        # with torch.no_grad():
        #     embeddings = self.clip_model.get_image_features(pixel_values=images)
        embeddings = images.mean(dim=(2, 3))
        # Apply custom model layers on the extracted embeddings
        batch_size = images.size(0)
        predictions = torch.zeros(batch_size, self.num_heads, self.num_classes).to(device)

        for idx in range(self.num_heads):
            y = self.model[idx](embeddings)
            predictions[:, idx, :] = y

        # Apply softmax to get probabilities for each class
        predictions = F.softmax(predictions, dim=-1)
        return predictions