import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

# Define the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Pretrained CLIP model name
clip_model_name = "openai/clip-vit-base-patch32"

class Clip(nn.Module):
    def __init__(self, num_classes, num_heads, embedding_dim=512):
        super().__init__()

        # Load CLIP model and processor
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

        # Custom layers for predictions
        self.model = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            ) for _ in range(num_heads)
        ]).to(device)

        self.num_heads = num_heads
        self.num_classes = num_classes

    def forward(self, x):
        if x.ndim == 3:  # If the batch is missing the channel dimension
            x = x.unsqueeze(1)  # Add the channel dimension (for single channel images)
        if x.shape[1] == 1:  # If grayscale, repeat to make it RGB
            x = x.repeat(1, 3, 1, 1)
        # Process the images with the CLIP processor (for CLIP-specific preprocessing)
        inputs = self.processor(images=x, return_tensors="pt").to(device)

        # Extract image features using the CLIP model
        embeddings = self.clip_model.get_image_features(**inputs)

        # Initialize a tensor to store the predictions
        pred = torch.zeros(x.shape[0], self.num_heads, self.num_classes).to(x.device)

        # Process each head and store the predictions
        for idx in range(self.num_heads):
            y = self.model[idx](embeddings)  # Pass through the model head
            pred[:, idx, :] = y

        # Apply softmax to get probabilities for each class
        pred = F.softmax(pred, dim=-1)
        return pred