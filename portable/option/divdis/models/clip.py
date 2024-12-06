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

    def forward(self, pixel_values):
        """
        Forward pass through the CLIP model and classification head.

        Args:
            pixel_values: Tensor of image features.
        """
        # Forward pass through CLIP model to get embeddings
        print(f"[DEBUG] pixel_values shape: {pixel_values.shape}")
        clip_outputs = self.clip_model(pixel_values=pixel_values)
        embeddings = clip_outputs[1]  # Assuming embeddings are at index 1
        print(f"[DEBUG] embeddings shape: {embeddings.shape}")

        # Forward pass through the classification head
        output = self.classification_head(embeddings)
        return output


class Clip(nn.Module):
    def __init__(self, num_classes, num_heads, embedding_dim=512):
        super().__init__()
        # Initialize components
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

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
        """
        Forward pass for the Clip model.

        Args:
            images: List of images (PIL Images or tensors).
        """
        # Preprocess images
        inputs = self.processor(images=images, return_tensors="pt", do_rescale=False)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        print(f"[DEBUG] Processor inputs keys: {inputs.keys()}")
        print(f"[DEBUG] Processor pixel_values shape: {inputs['pixel_values'].shape}")

        # Extract image features using CLIP
        with torch.no_grad():
            embeddings = self.clip_model.get_image_features(**inputs)
        print(f"[DEBUG] Embeddings shape after CLIP: {embeddings.shape}")

        # Process embeddings with classification heads
        batch_size = len(images)
        predictions = torch.zeros(batch_size, self.num_heads, self.num_classes, device=device)
        for idx in range(self.num_heads):
            predictions[:, idx, :] = self.model[idx](embeddings)

        # Apply softmax over the class dimension
        predictions = F.softmax(predictions, dim=-1)
        print(f"[DEBUG] Predictions shape: {predictions.shape}")
        return predictions



