import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Define the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Pretrained CLIP model name
clip_model_name = "openai/clip-vit-base-patch32"

class Clip(nn.Module):
    def __init__(self, num_classes, num_heads, embedding_dim=512):
        super().__init__()

        # Load CLIP model and processor
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

        # Custom layers for predictions
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

    def forward(self, images):
        # Ensure images are in the format expected by the processor
        if isinstance(images, list):
            # Convert the list of PIL images into a batch of images
            images = [image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(image) for image in images]

        # Preprocess the images (resize and normalize) and convert to tensor
        inputs = self.processor(images=images, return_tensors="pt", padding=True)

        # Move inputs to the same device as the model
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Extract image features using CLIP
        with torch.no_grad():
            embeddings = self.clip_model.get_image_features(**inputs)

        # Apply custom layers on the embeddings
        batch_size = images[0].size[0]  # Get the batch size from the first image
        predictions = torch.zeros(batch_size, self.num_heads, self.num_classes, device=device)
        for idx in range(self.num_heads):
            predictions[:, idx, :] = self.model[idx](embeddings)

        # Apply softmax over the class dimension
        predictions = F.softmax(predictions, dim=-1)
        return predictions


# Example usage:
# if __name__ == "__main__":
#     num_classes = 10  # Number of classes to predict
#     num_heads = 3     # Number of prediction heads

#     # Instantiate the model
#     clip_model = Clip(num_classes=num_classes, num_heads=num_heads)

#     # Example dummy image batch (create dummy white images with size 224x224)
#     dummy_images = [Image.new("RGB", (224, 224), color="white") for _ in range(4)]

#     # Forward pass through the model
#     outputs = clip_model(dummy_images)

#     # Print the output shape
#     print(outputs.shape)  # Expected shape: (batch_size, num_heads, num_classes)
