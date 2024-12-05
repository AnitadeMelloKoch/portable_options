import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToPILImage, ToTensor
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

    def extract_features(self, images):
        # Preprocess the images
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            embeddings = self.clip_model.get_image_features(**inputs)
        return embeddings

    def forward(self, images):
        print("Entered forward method")
        print("Original images shape:", images.shape if isinstance(images, torch.Tensor) else "Not a tensor")

        # Ensure images are converted to PIL format if necessary
        if isinstance(images, torch.Tensor):
            if len(images.shape) == 4:  # Assuming [batch_size, channels, height, width]
                images = [ToPILImage()(image) for image in images]
            else:
                raise ValueError("Expected 4D tensor (batch_size, channels, height, width) for images.")
        elif isinstance(images, list) and not all(isinstance(img, Image.Image) for img in images):
            raise ValueError("All images in the list must be PIL images.")

        # Extract embeddings
        embeddings = self.extract_features(images)

        # Apply custom layers on the embeddings
        batch_size = len(images)
        predictions = torch.zeros(batch_size, self.num_heads, self.num_classes, device=device)

        for idx in range(self.num_heads):
            predictions[:, idx, :] = self.model[idx](embeddings)

        # Apply softmax over the class dimension
        predictions = F.softmax(predictions, dim=-1)
        return predictions


# # Example usage
# if __name__ == "__main__":
#     # Number of classes and heads
#     num_classes = 10
#     num_heads = 3

#     # Create the model
#     clip_model = Clip(num_classes=num_classes, num_heads=num_heads)

#     # Dummy images as PIL
#     dummy_images_pil = [Image.new("RGB", (224, 224), color="white") for _ in range(4)]

#     # Dummy images as tensors
#     dummy_images_tensor = torch.stack([
#         ToTensor()(Image.new("RGB", (224, 224), color="white")) for _ in range(4)
#     ])

#     # Forward pass with PIL images
#     outputs_pil = clip_model(dummy_images_pil)
#     print("Output (PIL):", outputs_pil.shape)

#     # Forward pass with Tensor images
#     outputs_tensor = clip_model(dummy_images_tensor)
#     print("Output (Tensor):", outputs_tensor.shape)
