import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from torchvision.transforms import ToPILImage

# Define the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Pretrained CLIP model name
clip_model_name = "openai/clip-vit-base-patch32"

class Clip(nn.Module):
    def __init__(self, num_classes, num_heads):
        super().__init__()

        # Load CLIP model and processor
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

        # Dynamically determine the embedding size from CLIP
        dummy_input = torch.zeros(1, 3, 224, 224, device=device)
        with torch.no_grad():
            self.embedding_dim = self.clip_model.get_image_features(
                pixel_values=dummy_input
            ).shape[-1]

        # Custom layers for predictions
        self.model = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embedding_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            ) for _ in range(num_heads)
        ]).to(device)

        self.num_heads = num_heads
        self.num_classes = num_classes

    def forward(self, images):
        print("Input images shape:", images.shape)

        # Ensure input is [batch_size, 3, height, width]
        if images.ndim == 3:  # If missing channel dimension
            images = images.unsqueeze(1)  # Add channel dimension
        if images.shape[1] == 1:  # If grayscale, repeat to make RGB
            images = images.repeat(1, 3, 1, 1)
        print("Images after channel adjustment:", images.shape)

        # Convert tensor images to PIL format if needed
        if isinstance(images, torch.Tensor):
            # Ensure pixel values are in [0, 255]
            if images.dtype != torch.uint8:
                images = (images * 255).byte()
            images = [ToPILImage()(img) for img in images]

        print(f"Converted {len(images)} images to PIL format.")

        # Preprocess the images using CLIPProcessor
        inputs = self.processor(images=images, return_tensors="pt", do_rescale=False)
        print("Processor inputs:", {key: value.shape for key, value in inputs.items()})

        # Move inputs to the same device as the model
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Extract image embeddings from the CLIP model
        with torch.no_grad():
            embeddings = self.clip_model.get_image_features(**inputs)

        print("Embeddings shape:", embeddings.shape)

        # Initialize predictions tensor
        batch_size = embeddings.size(0)
        predictions = torch.zeros(batch_size, self.num_heads, self.num_classes, device=device)

        # Process embeddings through each head
        for idx in range(self.num_heads):
            predictions[:, idx, :] = self.model[idx](embeddings)
            print(f"Head {idx} predictions shape:", predictions[:, idx, :].shape)

        # Apply softmax over the class dimension
        predictions = F.softmax(predictions, dim=-1)
        print("Final predictions shape:", predictions.shape)
        return predictions
