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
        
        # Full model wrapping CLIP with classification heads
        self.full_model = nn.ModuleList([
            nn.Sequential(self.clip_model, classification_head) for classification_head in self.model
        ]).to(device)
        
        self.num_heads = num_heads
        self.num_classes = num_classes

    def forward(self, images):
        print("Entered forward pass")
        print("Images shape:", images.shape)
        
        # Preprocess the images, disabling rescale to avoid double normalization
        inputs = self.processor(images=images, return_tensors="pt", do_rescale=False)
        
        # Move inputs to the same device as the model
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        # Extract image features using CLIP
        with torch.no_grad():
            embeddings = self.clip_model.get_image_features(**inputs)
        
        # Apply the classification heads
        batch_size = embeddings.size(0)
        predictions = torch.zeros(batch_size, self.num_heads, self.num_classes, device=device)
        
        for idx in range(self.num_heads):
            predictions[:, idx, :] = self.full_model[idx][1](embeddings)  # Access the classification head
        
        # Apply softmax over the class dimension
        predictions = F.softmax(predictions, dim=-1)
        return predictions
