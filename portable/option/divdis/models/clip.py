import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np  # For handling image arrays if necessary

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

    def forward(self, images):
        # If images are a tensor, ensure they have the correct format (batch_size, 3, height, width)
        if isinstance(images, torch.Tensor):
            # Check if the tensor has 3 channels (RGB)
            if images.ndimension() == 3:
                # If grayscale (1 channel) or RGBA (4 channels), fix the format
                if images.size(0) == 1:  # Grayscale (1 channel)
                    images = images.repeat(3, 1, 1)  # Convert to 3 channels
                elif images.size(0) == 4:  # RGBA (4 channels)
                    images = images[:3, :, :]  # Keep only the first 3 channels (RGB)
            elif images.ndimension() == 4:
                # If the tensor has batch dimension, ensure the format is [batch_size, 3, height, width]
                if images.size(1) == 1:
                    images = images.repeat(1, 3, 1, 1)  # Convert to RGB
                elif images.size(1) == 4:
                    images = images[:, :3, :, :]  # Keep only the first 3 channels (RGB)

        # If images are a list of PIL images, ensure they are in RGB format
        elif isinstance(images, list):
            images = [img.convert("RGB") if isinstance(img, Image.Image) and img.mode != "RGB" else img for img in images]

        # Preprocess the images with `do_rescale=False` to avoid double rescaling
        inputs = self.processor(images=images, return_tensors="pt", do_rescale=False)

        # Ensure the tensor is on the correct device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Extract image features using CLIP
        with torch.no_grad():
            embeddings = self.clip_model.get_image_features(**inputs)

        # Apply custom layers on the embeddings
        batch_size = embeddings.size(0)  # Get the batch size from embeddings
        predictions = torch.zeros(batch_size, self.num_heads, self.num_classes, device=device)
        for idx in range(self.num_heads):
            predictions[:, idx, :] = self.model[idx](embeddings)

        # Apply softmax over the class dimension
        predictions = F.softmax(predictions, dim=-1)
        return predictions
