import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

# Define the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Pretrained CLIP model name
clip_model_name = "openai/clip-vit-base-patch32"

class Clip(nn.Module):
    def __init__(self, num_classes, num_heads, embedding_dim=512):
        super().__init__()
        # Load the vision model of CLIP
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).vision_model.to(device)

        # Linear layer to project embeddings (CLIP uses 768-dim features by default)
        self.projection = nn.Linear(768, embedding_dim).to(device)

        # Define classification heads
        self.model = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )
            for _ in range(num_heads)
        ]).to(device)

        self.full_model = nn.ModuleList([
            nn.Sequential(self.clip_model, classification_head)
            for classification_head in self.model
        ])
        self.num_heads = num_heads
        self.num_classes = num_classes

    def forward(self, images):
        """
        Args:
            images: Preprocessed images as a tensor of shape [batch_size, 3, H, W].
        Returns:
            predictions: Tensor of shape [batch_size, num_heads, num_classes] with probabilities.
        """
        # Pass the images through CLIP's vision model
        vision_outputs = self.clip_model(images)

        # Extract the last hidden state (patch embeddings) from vision_outputs
        last_hidden_state = vision_outputs[0]  # Shape: [batch_size, num_patches+1, 768]

        # Use the CLS token (first token) as a global representation
        cls_embeddings = last_hidden_state[:, 0, :]  # Shape: [batch_size, 768]

        # Project embeddings to a lower-dimensional space
        embeddings = self.projection(cls_embeddings)  # Shape: [batch_size, embedding_dim]

        # Initialize predictions tensor
        batch_size = embeddings.size(0)
        predictions = torch.zeros(batch_size, self.num_heads, self.num_classes, device=device)

        # Pass through each classification head
        for idx, head in enumerate(self.model):
            predictions[:, idx, :] = head(embeddings)

        # Apply softmax over the class dimension
        predictions = F.softmax(predictions, dim=-1)
        return predictions