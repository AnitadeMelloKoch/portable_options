import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

# Define the device
device = "cuda" if torch.cuda.is_available() else "cpu"

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
                nn.Linear(embedding_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),  # Corrected input size for the next layer
                nn.ReLU(),
                nn.Linear(128, num_classes)
            ) for _ in range(num_heads)
        ]).to(device)
        
        self.num_heads = num_heads
        self.num_classes = num_classes

    def forward(self, images):
        # Check if images are torch tensors; convert to PIL if necessary
        if isinstance(images, torch.Tensor):
            images = [transforms.ToPILImage()(img) for img in images]

        # Preprocess the images
        inputs = self.processor(
            images=images,
            return_tensors="pt",
            do_rescale=False  # Disable rescaling if input images are normalized
        ).to(device)

        # Extract image features using CLIP
        with torch.no_grad():
            embeddings = self.clip_model.get_image_features(**inputs)

        # Apply custom layers on the embeddings
        predictions = torch.zeros(len(images), self.num_heads, self.num_classes, device=device)
        for idx in range(self.num_heads):
            predictions[:, idx, :] = self.model[idx](embeddings)

        # Apply softmax over the class dimension
        predictions = F.softmax(predictions, dim=-1)
        return predictions
