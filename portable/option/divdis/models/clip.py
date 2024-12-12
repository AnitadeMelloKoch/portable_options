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

        # Global average pooling layer (to avoid manual mean reduction issues)
        self.global_pooling = nn.AdaptiveAvgPool1d(1)

        # Linear projection to 512 dimensions
        self.pooling = nn.Sequential(
            nn.Linear(768, 512),  # Assuming original embedding dimension is 768
            nn.ReLU()
        )

    def forward(self, images):
        # Preprocess images
        inputs = self.processor(images=images, return_tensors="pt", do_rescale=False)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # Ensure gradients are tracked for pixel values
        inputs['pixel_values'].requires_grad_(True)
        print(f"pixel_values requires_grad: {inputs['pixel_values'].requires_grad}")

        # Extract image features
        vision_outputs = self.clip_vision_model(pixel_values=inputs['pixel_values'])

        # Get last hidden states (shape: [batch_size, sequence_length, feature_dim])
        feature_map = vision_outputs.last_hidden_state
        print(f"feature_map requires_grad: {feature_map.requires_grad}")

        # Apply global average pooling over the sequence_length dimension
        feature_map = feature_map.permute(0, 2, 1)  # Change to [batch_size, feature_dim, sequence_length]
        pooled_output = self.global_pooling(feature_map)  # [batch_size, feature_dim, 1]
        pooled_output = pooled_output.squeeze(-1)  # [batch_size, feature_dim]

        # Pass through the pooling layer to project to 512 dimensions
        embeddings = self.pooling(pooled_output)  # Shape: [batch_size, 512]
        print(f"embeddings requires_grad: {embeddings.requires_grad}")
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