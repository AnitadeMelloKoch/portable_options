import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

# Define the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Pretrained CLIP model name
clip_model_name = "openai/clip-vit-base-patch32"

class PrintLayer(torch.nn.Module):
    # print input. For debugging
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        print(x.shape)
        return x

class GlobalAveragePooling2D(nn.Module):
    # Custom global average pooling layer
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean(dim=(2, 3))  # Average over height and width

class ClipVisionEmbedding(nn.Module):
    def __init__(self, clip_model_name, device):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_vision_model = CLIPModel.from_pretrained(clip_model_name).vision_model
        self.device = device

    def forward(self, images):
        # Preprocess images
        inputs = self.processor(images=images, return_tensors="pt", do_rescale=False)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # Extract image features
        with torch.no_grad():
            vision_outputs = self.clip_vision_model(pixel_values=inputs['pixel_values'])
        
        # Get the embeddings
        embeddings = vision_outputs.pooler_output  # Shape: [batch_size, embedding_dim]
        return embeddings

class Clip(nn.Module):
    def __init__(self, num_classes, num_heads, embedding_dim=512):
        super().__init__()

        # CLIP Vision Embedding Model
        self.embedding_class = ClipVisionEmbedding(clip_model_name, device)

        # Define classification heads
        self.model = nn.ModuleList([
            nn.Sequential(
                nn.LazyLinear(512),
                nn.LazyLinear(256),
                nn.LazyLinear(num_classes)
            )
            for _ in range(num_heads)
        ])

        self.num_heads = num_heads
        self.num_classes = num_classes

        # Full model includes embedding and classification head
        self.full_model = nn.ModuleList([
            nn.Sequential(
                PrintLayer(),
                self.embedding_class,
                PrintLayer(),
                GlobalAveragePooling2D(),
                PrintLayer(),
                classification_head,
                PrintLayer()
            )
            for classification_head in self.model
        ])

    def forward(self, x):
        print("x shape:", x.shape)

        # Prediction logic: apply embedding and classification layers
        pred = torch.zeros(len(x), self.num_heads, self.num_classes).to(x.device)
        for idx in range(self.num_heads):
            y = self.full_model[idx](x)
            print("y shape:", y.shape)
            pred[:, idx, :] = y

        # Apply softmax to get probabilities for each class
        pred = F.softmax(pred, dim=-1)
        return pred
