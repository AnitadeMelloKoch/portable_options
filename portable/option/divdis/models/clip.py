import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pretrained CLIP model name
clip_model_name = "openai/clip-vit-base-patch32"

class HeadedCLIPModel(nn.Module):
    """
    Combines the CLIP model with a custom classification head.
    """
    def __init__(self, clip_model, classification_head):
        super(HeadedCLIPModel, self).__init__()
        self.clip_model = clip_model
        self.classification_head = classification_head

    def forward(self, pixel_values, **kwargs):
        """
        Forward pass through the CLIP model and classification head.
        Args:
            pixel_values: Preprocessed input images for CLIP.
            **kwargs: Additional arguments for the CLIP model.
        Returns:
            Output from the classification head.
        """
        # Extract embeddings from the CLIP model
        clip_outputs = self.clip_model(pixel_values=pixel_values, **kwargs)
        embeddings = clip_outputs[1]  # Assumes embeddings are at index 1

        # Pass embeddings through the classification head
        return self.classification_head(embeddings)

class Clip(nn.Module):
    """
    Custom CLIP-based model with multiple classification heads.
    """
    def __init__(self, num_classes, num_heads, embedding_dim=512):
        super().__init__()
        # Load pretrained CLIP model and processor
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

        # Define multiple classification heads
        self.classification_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )
            for _ in range(num_heads)
        ]).to(device)

        # Combine CLIP model with each classification head
        self.full_model = nn.ModuleList([
            HeadedCLIPModel(self.clip_model, head) for head in self.classification_heads
        ]).to(device)

        self.num_heads = num_heads
        self.num_classes = num_classes

    def forward(self, images):
        """
        Forward pass for processing images through CLIP and classification heads.
        Args:
            images: List of input images (e.g., PIL.Image.Image).
        Returns:
            Predictions for each image from all classification heads.
        """
        # # Ensure images are resized to the expected dimensions
        # resized_images = [image.resize((224, 224)) for image in images]

        # Preprocess images with CLIPProcessor
        inputs = self.processor(images=images, return_tensors="pt", do_rescale=False)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Extract image embeddings using the CLIP model
        with torch.no_grad():
            embeddings = self.clip_model.get_image_features(**inputs)

        # Apply classification heads to embeddings
        batch_size = len(images)
        predictions = torch.zeros(batch_size, self.num_heads, self.num_classes, device=device)
        for idx, head in enumerate(self.classification_heads):
            predictions[:, idx, :] = head(embeddings)

        # Normalize predictions with softmax
        return F.softmax(predictions, dim=-1)
