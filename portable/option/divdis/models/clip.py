import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPVisionModel

# Define the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Pretrained CLIP model name
clip_model_name = "openai/clip-vit-base-patch32"

# PrintLayer for debugging shapes at various stages
class PrintLayer(torch.nn.Module):
    def __init__(self, label=""):
        super().__init__()
        self.label = label

    def forward(self, x):
        print(f"{self.label} shape: {x.shape}, requires_grad: {x.requires_grad}")
        return x

# Vision embedding class
class ClipVisionEmbedding(nn.Module):
    def __init__(self, clip_model_name, device):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_vision_model = CLIPVisionModel.from_pretrained(clip_model_name)
        self.device = device

        # Linear projection to 512 dimensions
        self.project_to_512 = nn.Linear(768, 512)

    def forward(self, images):
        # Preprocess images
        inputs = self.processor(images=images, return_tensors="pt", do_rescale=False)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # Ensure gradients are tracked
        inputs['pixel_values'].requires_grad_(True)
        print(f"pixel_values.requires_grad: {inputs['pixel_values'].requires_grad}")

        # Enable gradient tracking explicitly
        with torch.set_grad_enabled(True):
            vision_outputs = self.clip_vision_model(pixel_values=inputs['pixel_values'])

        # Extract CLS token
        cls_embedding = vision_outputs.last_hidden_state[:, 0, :]
        print(f"cls_embedding.requires_grad (pre-projection): {cls_embedding.requires_grad}")

        # Project to 512 dimensions
        embeddings = self.project_to_512(cls_embedding)
        embeddings.requires_grad_(True)  # Ensure projection output tracks gradients
        print(f"embeddings.requires_grad: {embeddings.requires_grad}")

        # Add hook to check gradients during backward pass
        embeddings.register_hook(lambda grad: print(f"Embeddings Grad: {grad}"))

        return embeddings

# Main CLIP-based multi-head classifier
class Clip(nn.Module):
    def __init__(self, num_classes, num_heads, embedding_dim=512):
        super().__init__()
        
        # Vision embedding module
        self.clip_embedding = ClipVisionEmbedding(clip_model_name, device).to(device)
        
        # Classification heads
        self.model = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            ) for _ in range(num_heads)
        ]).to(device)
        
        # Debugging full pipeline
        self.full_model = nn.ModuleList([
            nn.Sequential(
                PrintLayer(label="Input"),
                self.clip_embedding,
                PrintLayer(label="After Embedding"),
                classification_head,
                PrintLayer(label="After Classification")
            ) for classification_head in self.model
        ])

        self.num_heads = num_heads
        self.num_classes = num_classes

    def forward(self, x):
        print("x shape:", x.shape)
        pred = torch.zeros(len(x), self.num_heads, self.num_classes).to(device)

        for idx in range(self.num_heads):
            # Forward pass through embedding and classification head
            y = self.full_model[idx](x)
            print(f"y shape for head {idx}: {y.shape}")
            pred[:, idx, :] = y

        # Apply softmax
        pred = F.softmax(pred, dim=-1)
        return pred

