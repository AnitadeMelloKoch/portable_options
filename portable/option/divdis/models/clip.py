import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from torchvision import transforms  # For handling image transformations
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
        # Verify and preprocess images
        print("proproces:",images.shape)
        # Ensure the image input is in the correct format
        if images.ndim == 3:  # Single image, needs to be wrapped in a list
            images = [images]  # Convert to list
        elif images.ndim == 4:  # Already a batch of images
            images = images.squeeze(0)  # Remove batch dimension if any (optional)

        # Check and enforce exactly 4 channels for each image
        for i in range(len(images)):
            if images[i].shape[0] == 1:  # If grayscale (1 channel), repeat to form 4 channels
                images[i] = images[i].repeat(3, 1, 1)
            elif images[i].shape[0] == 4:  # If RGB (3 channels), repeat to form 4 channels
                images[i] = images[i].repeat(3, 1, 1)
            elif images[i].shape[0] == 3:  # Already 4 channels, do nothing
                pass
            else:  # If more than 3 channels, slice to keep the first 3 channels
                images[i] = images[i][:3, :, :]


        print("after:", images.shape)
        
        
        # Preprocess the images with `do_rescale=False` to avoid double rescaling
        inputs = self.processor(images=images, return_tensors="pt", do_rescale=False)

        # Move inputs to the same device as the model
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

