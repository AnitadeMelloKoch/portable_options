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
    def __init__(self, num_classes, num_heads, embedding_dim=512, saved_embedding_path="portable/option/divdis/models/clip_embeddings.pt"):
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
        self.clip_embedding = self._load_embeddings(saved_embedding_path).to(self.device)
        
        # Define classification heads
        self.model = nn.ModuleList([  # Define each head
            nn.Sequential(
                nn.Linear(embedding_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            ) for _ in range(num_heads)
        ]).to(device)

        # Full model for visualization: embeddings -> classification heads
        self.full_model = nn.ModuleList([  # Define the full model for each head
            nn.Sequential(
                PrintLayer(),
                nn.Identity(),  # Placeholder to simulate embedding input
                PrintLayer(),
                head,
                PrintLayer()
            ) for head in self.model
        ])
        
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

        # Ensure indices are on the correct device and cast to long type
        x = x.to(self.device).long()  # Ensure x is of type long
        batch_size = x.shape[0]

        # Select the embeddings corresponding to input indices
        selected_embeddings = self.clip_embedding[x]  # Shape: [batch_size, embedding_dim]
        print(f"Selected embeddings shape: {selected_embeddings.shape}")

        # Output tensor to store predictions
        pred = torch.zeros(batch_size, self.num_heads, self.num_classes).to(self.device)

        # Forward pass through each classification head
        for idx in range(self.num_heads):
            y = self.full_model[idx](selected_embeddings)  # Pass selected embeddings
            print(f"Shape of y for head {idx}: {y.shape}")

            # Check if the tensor has more than 2 dimensions, and flatten it if necessary
            if y.dim() > 2:  # This includes 3D, 4D, or 5D tensors
                # If it is a 3D tensor like [batch_size, channels, spatial_size]
                # or a tensor with flattened dimensions like [batch_size, feature_map_size, num_classes]
                # we will flatten all spatial/feature map dimensions (if any)
                y = y.view(batch_size, -1, self.num_classes)  # Flatten feature dimensions
                print(f"Shape of y after flattening for head {idx}: {y.shape}")
            elif y.dim() == 2:  # If already 2D (e.g., [batch_size, num_classes])
                pass  # No need to change the shape

            # Ensure y has the correct shape for the classification head
            if y.dim() == 2 and y.shape[0] == batch_size:
                pred[:, idx, :] = y
            else:
                raise RuntimeError(f"Unexpected tensor shape: {y.shape}")

        # Apply softmax to get probabilities
        pred = F.softmax(pred, dim=-1)
        return pred
