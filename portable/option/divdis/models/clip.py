import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel
from torchvision import transforms
from PIL import Image

class CLIPEnsemble(nn.Module):
    def __init__(self, num_classes, num_heads, embedding_dim=512):
        super().__init__()

        # Load the pretrained CLIP model
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda" if torch.cuda.is_available() else "cpu")

        # Define custom model layers for each head
        self.model = nn.ModuleList([
            nn.Sequential(
                nn.LazyLinear(1000),  # You can adjust the dimensions here as needed
                nn.LazyLinear(700),
                nn.LazyLinear(num_classes)  # Final output layer for classification
            ) for _ in range(num_heads)
        ])
        
        self.num_heads = num_heads
        self.num_classes = num_classes

        # Image preprocessing pipeline (resize, crop, normalize)
        self.preprocess = transforms.Compose([
            transforms.Resize(224),          # Resize to 224x224 (instead of 256x256)
            transforms.CenterCrop(224),      # Crop to 224x224
            transforms.ToTensor(),           # Convert to tensor
            transforms.Normalize(            # Normalize using the CLIP standard mean and std
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
        ])

    def forward(self, images):
        # If images are a list of PIL images, preprocess them
        if isinstance(images, list):
            images = [self.preprocess(img).unsqueeze(0) for img in images]  # Preprocess each image
            images = torch.cat(images, dim=0)  # Stack the images into a batch

        # Ensure images have the correct shape [batch_size, 3, height, width]
        if images.dim() == 3:  # If missing batch dimension
            images = images.unsqueeze(0)  # Add batch dimension

        # Move the images to the same device as the model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        images = images.to(device)

        # Extract image features using CLIP (image embeddings)
        with torch.no_grad():
            embeddings = self.clip_model.get_image_features(pixel_values=images)

        # Apply custom model layers on the extracted embeddings
        batch_size = images.size(0)
        predictions = torch.zeros(batch_size, self.num_heads, self.num_classes).to(device)

        for idx in range(self.num_heads):
            predictions[:, idx, :] = self.model[idx](embeddings)

        # Apply softmax to get probabilities for each class
        predictions = F.softmax(predictions, dim=-1)
        return predictions
