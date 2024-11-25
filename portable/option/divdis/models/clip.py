import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from torchvision.transforms import ToPILImage

# Define the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Pretrained CLIP model name
clip_model_name = "openai/clip-vit-base-patch32"

class Clip(nn.Module):
    def __init__(self, num_classes, num_heads, embedding_dim=224):  # Set embedding_dim to 224
        super().__init__()

        # Load CLIP model and processor
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

        # Custom layers for predictions
        self.model = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, 128),  # Adjust input size to 224
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            ) for _ in range(num_heads)
        ]).to(device)

        self.num_heads = num_heads
        self.num_classes = num_classes

    def forward(self, images):
        # Convert tensor images to PIL if necessary
        if isinstance(images, torch.Tensor):
            # Ensure pixel values are in [0, 255]
            if images.dtype != torch.uint8:
                images = (images * 255).byte()
            images = [ToPILImage()(img) for img in images]

        # Preprocess the images
        inputs = self.processor(images=images, return_tensors="pt", do_rescale=False)

        # Move inputs to the same device as the model
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Extract image features using CLIP
        with torch.no_grad():
            embeddings = self.clip_model.get_image_features(**inputs)

        # Reduce embedding space to 224
        if embeddings.size(-1) != 224:  # Check if reduction is needed
            reduction_layer = nn.Linear(embeddings.size(-1), 224).to(device)
            embeddings = reduction_layer(embeddings)

        # Apply custom layers on the embeddings
        batch_size = len(images) if isinstance(images, list) else images.size(0)
        predictions = torch.zeros(batch_size, self.num_heads, self.num_classes, device=device)
        for idx in range(self.num_heads):
            predictions[:, idx, :] = self.model[idx](embeddings)

        # Apply softmax over the class dimension
        predictions = F.softmax(predictions, dim=-1)
        return predictions
