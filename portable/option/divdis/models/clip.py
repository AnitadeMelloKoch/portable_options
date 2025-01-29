import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPVisionModel

# Set up the device (ensure consistency across the model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Pretrained CLIP model name
clip_model_name = "openai/clip-vit-base-patch32"

class PrintLayer(torch.nn.Module):
    # Print input. For debugging
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        print(f"PrintLayer - Tensor shape: {x.shape} | Device: {x.device}")
        return x


class ClipVisionEmbedding(nn.Module):
    def __init__(self, clip_model_name, device):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_vision_model = CLIPVisionModel.from_pretrained(clip_model_name).to(device)  # Move model to device
        self.device = device

        # Linear projection directly to 512 dimensions
        self.project_to_512 = nn.Linear(768, 512).to(device)  # Ensure projection is on the correct device

    def forward(self, images):
        if not isinstance(images, torch.Tensor):
            raise ValueError("Images must be a torch.Tensor type.")
        images = images.to(self.device)  # Ensure input is on the correct device

        print(f"Input shape: {images.shape}, requires_grad: {images.requires_grad}, Device: {images.device}")

        inputs = {'pixel_values': images}
        inputs['pixel_values'].requires_grad_(True)

        print(f"Pixel values requires_grad: {inputs['pixel_values'].requires_grad} | Device: {inputs['pixel_values'].device}")

        with torch.enable_grad():
            vision_outputs = self.clip_vision_model(pixel_values=inputs['pixel_values'])
            print(f"vision_outputs.shape: {vision_outputs.last_hidden_state.shape} | Device: {vision_outputs.last_hidden_state.device}")

        cls_embedding = vision_outputs.last_hidden_state[:, 0, :]
        cls_embedding = cls_embedding.to(self.device)  # Move embedding to the correct device

        print(f"cls_embedding.requires_grad (pre-projection): {cls_embedding.requires_grad} | Device: {cls_embedding.device}")

        # Project to 512 dimensions
        embeddings = self.project_to_512(cls_embedding)
        embeddings = embeddings.to(self.device)

        print(f"embeddings.requires_grad: {embeddings.requires_grad} | Device: {embeddings.device}")
        return embeddings


class Clip(nn.Module):
    def __init__(self, num_classes, num_heads, embedding_dim=768):
        super().__init__()
        
        # Ensure the CLIP embedding model is on the correct device
        self.clip_embedding = ClipVisionEmbedding(clip_model_name, device).to(device)

        # Define classification heads
        self.model = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, 128).to(device),
                nn.ReLU(),
                nn.Linear(128, 64).to(device),
                nn.ReLU(),
                nn.Linear(64, num_classes).to(device)
            ) for _ in range(num_heads)
        ])

        # Keep the full model for debugging
        self.full_model = nn.ModuleList([
            nn.Sequential(
                PrintLayer().to(device),
                self.clip_embedding,
                PrintLayer().to(device),
                classification_head,
                PrintLayer().to(device)
            ) for classification_head in self.model
        ])
        
        self.num_heads = num_heads
        self.num_classes = num_classes

    def forward(self, x):
        print(f"Original x shape: {x.shape} | Device: {x.device}")

        # x = x.to(device)  # Move input tensor to correct device
        x = x.unsqueeze(1).repeat(1, 3, 1, 1).to(device)  # Ensure correct shape: [10, 3, 768, 768]

        print(f"Modified x shape: {x.shape} | Device: {x.device}")

        pred = torch.zeros(len(x), self.num_heads, self.num_classes, device=device)  # Ensure tensor is on the same device

        for idx in range(self.num_heads):
            y = self.full_model[idx](x)  # Forward pass
            print(f"Output shape for head {idx}: {y.shape} | Device: {y.device}")
            pred[:, idx, :] = y.to(device)

        pred = F.softmax(pred, dim=-1)  # Apply softmax
        return pred

