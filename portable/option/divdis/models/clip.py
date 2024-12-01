import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel
from PIL import Image
from torchvision import transforms

# Define the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Pretrained CLIP model name
clip_model_name = "openai/clip-vit-base-patch32"

class Clip(nn.Module):
    def __init__(self, num_classes, num_heads, embedding_dim=512):
        super().__init__()

        # Load CLIP model
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)

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

        # Define the image preprocessing pipeline (resize, center crop, normalize)
        self.preprocess = transforms.Compose([
            transforms.Resize(256),          # Resize to 256x256
            transforms.CenterCrop(224),      # Crop to 224x224
            transforms.ToTensor(),           # Convert to tensor
            transforms.Normalize(            # Normalize using the same mean and std as CLIP
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
        ])

    def forward(self, images):
        # If images are a list of PIL images, apply preprocessing
        if isinstance(images, list):
            images = [self.preprocess(img).unsqueeze(0) for img in images]  # Preprocess each image
            images = torch.cat(images, dim=0)  # Stack the images into a batch

        # Move the images to the same device as the model
        images = images.to(device)

        # Extract image features using CLIP (image embeddings)
        with torch.no_grad():
            embeddings = self.clip_model.get_image_features(pixel_values=images)

        # Apply custom layers on the embeddings
        batch_size = images.size(0)
        predictions = torch.zeros(batch_size, self.num_heads, self.num_classes, device=device)
        for idx in range(self.num_heads):
            predictions[:, idx, :] = self.model[idx](embeddings)

        # Apply softmax over the class dimension
        predictions = F.softmax(predictions, dim=-1)
        return predictions


