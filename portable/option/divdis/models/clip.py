import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPVisionModel

# Define the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Pretrained CLIP model name
clip_model_name = "openai/clip-vit-base-patch32"

class PrintLayer(nn.Module):
    """Print input tensor shape for debugging."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(f"Shape: {x.shape}")
        return x

class Clip(nn.Module):
    def __init__(self, num_classes, num_heads, embedding_dim=768, saved_embedding_path="portable/option/divdis/models/clip_embeddings.pt"):
        """
        Clip model with classification heads using preloaded embeddings.

        Args:
            num_classes (int): Number of output classes.
            num_heads (int): Number of independent classification heads.
            embedding_dim (int): Dimension of the embeddings.
            saved_embedding_path (str): Path to precomputed embeddings file (.pt).
        """
        super().__init__()
        self.device = device
        
        # Load precomputed embeddings
        self.clip_embedding = self._load_embeddings(saved_embedding_path).requires_grad_(True).to(self.device)
        
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
        
        self.num_heads = num_heads
        self.num_classes = num_classes

    def _load_embeddings(self, path):
        """
        Load precomputed embeddings directly from a .pt file.

        Args:
            path (str): Path to the precomputed embeddings file.

        Returns:
            torch.Tensor: Loaded embeddings tensor.
        """
        if path.endswith(".pt"):
            embeddings = torch.load(path)
            if not isinstance(embeddings, torch.Tensor):
                raise ValueError("Loaded embeddings are not a torch.Tensor.")
            print(f"Loaded embeddings with shape: {embeddings.shape}")
            return embeddings
        else:
            raise ValueError("Unsupported embedding file format. Only .pt files are supported.")

    def forward(self, x):
        """
        Forward pass through the full model (embeddings + classification).
        
        Args:
            x (torch.Tensor): Indices for selecting embeddings.
        
        Returns:
            torch.Tensor: Output predictions with shape [batch_size, num_heads, num_classes].
        """
        print("Input shape:", x.shape)
        # Ensure indices are on the correct device
        x = x.to(self.device)
        batch_size = x.shape[0]
        
        # Forward pass through full model (embedding + classification)
        pred = torch.zeros(batch_size, self.num_heads, self.num_classes).to(self.device)
        
        for idx in range(self.num_heads):
            # Pass preloaded embeddings to the classification head
            y = self.model[idx](self.clip_embedding)[:batch_size]
            
            # Debugging shapes
            print(f"Shape of y for head {idx}: {y.shape}")
            
            # Ensure y has the correct shape
            if y.shape == (batch_size, self.num_classes):
                pred[:, idx, :] = y
            else:
                raise RuntimeError(f"Shape mismatch: expected ({batch_size}, {self.num_classes}), got {y.shape}")
        
        # Apply softmax to get probabilities
        pred = F.softmax(pred, dim=-1)
        return pred
