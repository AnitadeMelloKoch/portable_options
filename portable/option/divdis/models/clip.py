import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
# Define the device
device = "cuda" if torch.cuda.is_available() else "cpu"
# Pretrained CLIP model name
clip_model_name = "openai/clip-vit-base-patch32"


class HeadedCLIPModel(nn.Module):
    def __init__(self, clip_model, classification_head):
        super(HeadedCLIPModel, self).__init__()
        self.clip_model = clip_model
        self.classification_head = classification_head

    def forward(self, pixel_values, **kwargs):
        # Forward pass through CLIP model to get embeddings
        clip_outputs = self.clip_model(pixel_values=pixel_values, **kwargs)
        embeddings = clip_outputs[1]  # Assuming embeddings are at index 1

        # Forward pass through the classification head
        output = self.classification_head(embeddings)
        return output
    
class Clip(nn.Module):
    def __init__(self, num_classes, num_heads, embedding_dim=512):
        super().__init__()
        # Initialize components
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

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

        # Create individual models for each head
        self.full_model = nn.ModuleList([
            HeadedCLIPModel(self.clip_model, head) for head in self.model
        ]).to(device)
        self.num_heads = num_heads
        self.num_classes = num_classes

    def forward(self, images):
        # Ensure images are preprocessed to match expected input format
        inputs = self.processor(images=images, return_tensors="pt", do_rescale=False)
        
        # Move inputs to the same device as the model
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Ensure we only pass image-related keys to the clip model
        image_inputs = {key: value for key, value in inputs.items() if key != 'input_ids'}

        # Extract image features using CLIP
        with torch.no_grad():
            embeddings = self.clip_model.get_image_features(**image_inputs)

        # Apply custom layers on the embeddings
        batch_size = images.size(0)
        predictions = torch.zeros(batch_size, self.num_heads, self.num_classes, device=device)
        for idx in range(self.num_heads):
            predictions[:, idx, :] = self.model[idx](embeddings)
        
        # Apply softmax over the class dimension
        predictions = F.softmax(predictions, dim=-1)
        return predictions
