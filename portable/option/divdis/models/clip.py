import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np

# Define the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Pretrained CLIP model name
clip_model_name = "openai/clip-vit-base-patch32"

class Clip(nn.Module):
    def __init__(self, num_classes, num_heads, embedding_dim=512):
        super(Clip, self).__init__()
        
        # Load CLIP model and processor
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        # Custom classification heads for each head
        self.model = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            ) for _ in range(num_heads)
        ]).to(device)

        # Wrap the CLIP model with the custom heads
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.full_model = nn.ModuleList([
            nn.Sequential(self.clip_model, classfication_head) 
            for classfication_head in self.model
        ]).to(device)

    def forward(self, images):
        print("Entered forward pass")
        print("Images shape:", images.shape)
        
        # Preprocess the images using CLIPProcessor
        inputs = self.processor(images=images, return_tensors="pt")
        
        # Move inputs to the same device as the model
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        # Extract image features using CLIP model
        with torch.no_grad():
            embeddings = self.clip_model.get_image_features(**inputs)
            print("Embeddings shape:", embeddings.shape)

        # Initialize tensor to hold predictions
        batch_size = images.size(0)
        predictions = torch.zeros(batch_size, self.num_heads, self.num_classes, device=device)

        # Apply the classification heads to the embeddings
        for idx in range(self.num_heads):
            print(f"Prediction for head {idx}:")
            head_output = self.model[idx](embeddings)
            print(f"Head {idx} output shape:", head_output.shape)
            predictions[:, idx, :] = head_output
        
        # Apply softmax over the class dimension
        predictions = F.softmax(predictions, dim=-1)
        return predictions