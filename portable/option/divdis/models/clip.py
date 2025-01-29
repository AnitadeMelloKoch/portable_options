import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPVisionModel

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pretrained CLIP model name
clip_model_name = "openai/clip-vit-base-patch32"

class PrintLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        print(f"PrintLayer - Tensor shape: {x.shape} | Device: {x.device}")
        return x

class ClipVisionEmbedding(nn.Module):
    def __init__(self, clip_model_name, device):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_vision_model = CLIPVisionModel.from_pretrained(clip_model_name).to(device)  # Ensure model is on the right device
        self.device = device

    def forward(self, images):
        if not isinstance(images, torch.Tensor):
            raise ValueError("Images must be torch.Tensor type.")
        
        images = images.to(self.device)  # Move input tensor to the correct device
        images.requires_grad_(True)
        
        print(f"Input shape: {images.shape}, requires_grad: {images.requires_grad}, Device: {images.device}")
        
        inputs = {'pixel_values': images.to(self.device)}  # Ensure tensor is on the correct device
        inputs['pixel_values'].requires_grad_(True)
        
        print(f"Pixel values requires_grad: {inputs['pixel_values'].requires_grad} | Device: {inputs['pixel_values'].device}")
        
        with torch.enable_grad():
            vision_outputs = self.clip_vision_model(pixel_values=inputs['pixel_values'])
            print(f"vision_outputs.shape: {vision_outputs.last_hidden_state.shape} | Device: {vision_outputs.last_hidden_state.device}")
        
        cls_embedding = vision_outputs.last_hidden_state[:, 0, :]
        return cls_embedding

class Clip(nn.Module):
    def __init__(self, num_classes, num_heads, embedding_dim=768):
        super().__init__()
        self.clip_embedding = ClipVisionEmbedding(clip_model_name, device).to(device)
        self.model = nn.ModuleList([ 
            nn.Sequential(
                nn.Linear(embedding_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            ) for _ in range(num_heads)
        ]).to(device)

    def forward(self, x):
        print(f"Original x shape: {x.shape} | Device: {x.device}")
        
        x = x.to(device)  # Move input to the correct device
        x = x.unsqueeze(1).repeat(1, 3, 1, 1).to(device)  # Convert grayscale to 3-channel RGB
        print(f"Modified x shape: {x.shape} | Device: {x.device}")

        pred = torch.zeros(len(x), self.num_heads, self.num_classes, device=device)
        for idx in range(self.num_heads):
            y = self.model[idx](self.clip_embedding(x))
            print(f"y shape for head {idx}: {y.shape} | Device: {y.device}")
            pred[:, idx, :] = y
        
        return F.softmax(pred, dim=-1)
