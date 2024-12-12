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
        # Load only the vision part of the CLIP model
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).vision_model.to(device)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name).to(device)

        # Define classification heads
        self.model = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            ) for _ in range(num_heads)
        ]).to(device)

        self.full_model = nn.ModuleList([
            nn.Sequential(self.clip_model, classification_head)
            for classification_head in self.model
        ])
        # self.full_model = nn.ModuleList([nn.Sequential([self.clip_model,classfication_head]) for classfication_head in self.model]).to(device)
        self.num_heads = num_heads
        self.num_classes = num_classes

    def forward(self, images):
        # Ensure images are preprocessed to match expected input format
        inputs = self.processor(images=images, return_tensors="pt", do_rescale=False)
        
        # Move inputs to the same device as the model
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Extract image features using the CLIP vision model
        with torch.no_grad():
            vision_outputs = self.clip_model(pixel_values=inputs['pixel_values']).to(device)

        # Get the image embeddings
        embeddings = vision_outputs.pooler_output  # Shape: [batch_size, embedding_dim]

        # Apply custom layers on the embeddings
        batch_size = embeddings.size(0)
        predictions = torch.zeros(batch_size, self.num_heads, self.num_classes, device=device)
        for idx in range(self.num_heads):
            predictions[:, idx, :] = self.model[idx](embeddings)

        # Apply softmax over the class dimension
        predictions = F.softmax(predictions, dim=-1)
        return predictions