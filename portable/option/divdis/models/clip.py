import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

# Define the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Pretrained CLIP model name
clip_model_name = "openai/clip-vit-base-patch32"

class PrintLayer(torch.nn.Module):
    # Print input. For debugging
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        print(x.shape)
        return x

class ClipVisionEmbedding(nn.Module):
    def __init__(self, clip_model_name, device):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_vision_model = CLIPModel.from_pretrained(clip_model_name).vision_model
        self.device = device
        self.pooling = nn.Linear(768, 512)

    def forward(self, images):
        inputs = self.processor(images=images, return_tensors="pt", do_rescale=False)
        inputs['pixel_values'] = inputs['pixel_values'].to(self.device).requires_grad_(True)

        # Forward pass without no_grad
        vision_outputs = self.clip_vision_model(pixel_values=inputs['pixel_values'])
        feature_map = vision_outputs.last_hidden_state.mean(dim=1)  # Global Average Pooling
        embeddings = self.pooling(feature_map)
        
        print("embeddings requires_grad:", embeddings.requires_grad)
        return embeddings



class Clip(nn.Module):
    def __init__(self, num_classes, num_heads, embedding_dim=512):
        super().__init__()
        
        # Define the CLIP vision embedding module
        self.clip_embedding = ClipVisionEmbedding(clip_model_name, device).to(device)
        
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
        
        # Keep the full model for visualization
        self.full_model = nn.ModuleList([ 
            nn.Sequential(
                PrintLayer(),
                self.clip_embedding,
                PrintLayer(),
                classification_head,
                PrintLayer()
            ) for classification_head in self.model
        ])
        
        self.num_heads = num_heads
        self.num_classes = num_classes

    def forward(self, x):
        print("x shape:", x.shape)
        # Forward pass through full model (embedding + classification)
        pred = torch.zeros(len(x), self.num_heads, self.num_classes).to(device)
        
        for idx in range(self.num_heads):
            # Ensure gradients are computed in the entire flow
            y = self.full_model[idx](x)  # x -> CLIPEmbedding -> Classification head
            print("y shape:", y.shape)
            pred[:, idx, :] = y
        
        # Apply softmax to get probabilities
        pred = F.softmax(pred, dim=-1)
        return pred