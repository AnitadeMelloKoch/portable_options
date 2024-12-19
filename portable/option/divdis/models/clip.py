import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPVisionModel

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


class PrecomputedEmbeddings(nn.Module):
    """
    Wrapper module to load and retrieve precomputed embeddings.
    Acts as a replacement for ClipVisionEmbedding.
    """
    def __init__(self, embedding_path, device):
        """
        Args:
            embedding_path (str): Path to the precomputed embeddings file (.pt or .npy).
            device (str): Device to load embeddings on ("cpu" or "cuda").
        """
        super().__init__()
        self.embeddings = self._load_embeddings(embedding_path).to(device)
        print(f"Precomputed embeddings loaded. Shape: {self.embeddings.shape}")
        
    @staticmethod
    def _load_embeddings(path):
        """Load embeddings from a .pt or .npy file."""
        if path.endswith(".pt"):
            return torch.load(path)
        else:
            raise ValueError("Unsupported embedding file format. Use .pt or .npy.")
        
    def forward(self, indices):
        """
        Forward method to retrieve embeddings based on input indices.
        
        Args:
            indices (torch.Tensor): Indices to select embeddings from the saved space.
        
        Returns:
            torch.Tensor: Selected embeddings with shape [batch_size, embedding_dim].
        """
        print(f"Retrieving embeddings for indices: {indices}")
        return self.embeddings[indices]


class Clip(nn.Module):
    def __init__(self, num_classes, num_heads, embedding_dim=512, saved_embedding_path="portable/option/divdis/models/clip_embeddings.pt"):
        """
        Clip model with classification heads using precomputed embeddings.
        
        Args:
            num_classes (int): Number of output classes.
            num_heads (int): Number of independent classification heads.
            embedding_dim (int): Dimension of the input embeddings.
            saved_embedding_path (str): Path to precomputed embeddings (.pt or .npy).
            device (str): Device for computation ("cpu" or "cuda").
        """
        super().__init__()
        self.device = device
        
        # Use PrecomputedEmbeddings in place of ClipVisionEmbedding
        self.clip_embedding = PrecomputedEmbeddings(saved_embedding_path, device).to(device)
        
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
        
        # Full model for visualization (embedding -> classification)
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
        """
        Forward pass through the full model (embeddings + classification).
        
        Args:
            x (torch.Tensor): Indices for selecting embeddings.
        
        Returns:
            torch.Tensor: Output predictions with shape [batch_size, num_heads, num_classes].
        """
        print("Input shape:", x.shape)
        # Ensure indices are on the correct device
        x = x.to(self.device).long()
        
        # Forward pass through full model (embedding + classification)
        pred = torch.zeros(len(x), self.num_heads, self.num_classes).to(self.device)
        
        for idx in range(self.num_heads):
            y = self.full_model[idx](x)  # x -> PrecomputedEmbeddings -> Classification head
            print(f"Output shape for head {idx}: {y.shape}")
            pred[:, idx, :] = y
        
        # Apply softmax to get probabilities
        pred = F.softmax(pred, dim=-1)
        return pred
