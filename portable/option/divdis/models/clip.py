import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
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
        self.full_model = nn.ModuleList([
            nn.Sequential(self.clip_model, classification_head)
            for classification_head in self.model
        ])
        self.num_heads = num_heads
        self.num_classes = num_classes
        
    def forward(self, x):
        print("entered")
        print("images shape:", len(x))  # `images` is a list, so print its length
        
        # Preprocess the images
        inputs = self.processor(images=x, return_tensors="pt", do_rescale=False)  # Prevent double rescaling
        
        # Move inputs to the same device as the model
        inputs = {key: value.to(device) for key, value in inputs.items()}
        pixel_values = inputs['pixel_values']  # Explicitly extract `pixel_values`

        print("pixel_values shape:", pixel_values.shape)
        
        # Extract image features using CLIP
        with torch.no_grad():
            embeddings = self.clip_model.get_image_features(pixel_values=pixel_values)
            print("embeddings shape:", embeddings.shape)
        
        # Apply custom layers on the embeddings
        batch_size = pixel_values.size(0)
        predictions = torch.zeros(batch_size, self.num_heads, self.num_classes, device=device)
        
        for idx in range(self.num_heads):
            output = self.model[idx](embeddings)
            print(f"model output shape for head {idx}:", output.shape)
            predictions[:, idx, :] = output
        
        # Apply softmax over the class dimension
        predictions = F.softmax(predictions, dim=-1)
        return predictions

# # Example usage:
# if __name__ == "__main__":
#     # Number of classes and heads
#     num_classes = 10
#     num_heads = 3
#     # Create the model
#     clip_model = Clip(num_classes=num_classes, num_heads=num_heads)
#     # Dummy image batch (e.g., PIL images or similar)
#     from PIL import Image
#     dummy_images = [Image.new("RGB", (224, 224), color="white") for _ in range(4)]
#     # Forward pass
#     outputs = clip_model(dummy_images)
#     print(outputs.shape)  # Expected shape: (batch_size, num_heads, num_classes)