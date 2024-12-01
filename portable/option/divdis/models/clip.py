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
        # Check if images are in the correct format (list of PIL images or tensor)
        if isinstance(images, list):
            # Convert list of images to PIL if necessary (only if not already in PIL format)
            images = [Image.fromarray(img) if isinstance(img, np.ndarray) else img for img in images]

        # Ensure all images are in RGB format (3 channels)
        images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]

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
