import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

# Define the device
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model_name = "openai/clip-vit-base-patch32"

class Clip(nn.Module):
    def __init__(self, num_classes, num_heads):
        super().__init__()

        # Set embedding_class to use CLIP pre-trained weights
        self.embedding_class = CLIPModel.from_pretrained(clip_model_name).to(device)

        # Custom convolutional and fully connected layers
        self.model = nn.ModuleList([nn.Sequential(
                nn.LazyLinear(700),
                nn.LazyLinear(500),
                nn.LazyLinear(num_classes)
        ) for _ in range(num_heads)])
        
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.full_model = nn.ModuleList([
            nn.Sequential(self.clip_model, classification_head)
            for classification_head in self.model
        ])

    def forward(self, x, logits=False):
        # Get embeddings from CLIP model
        embedding = self.embedding_class.get_image_features(x)
        
        # Prediction logic: apply your custom model layers on the embedding
        pred = torch.zeros(x.shape[0], self.num_heads, self.num_classes).to(x.device)
        for idx in range(self.num_heads):
            y = self.model[idx](embedding)
            # print("y:",y.shape)
            pred[:, idx, :] = y
        
        if logits:
            return pred
        else:
            return F.softmax(pred, dim=-1)