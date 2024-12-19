import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPVisionModel
import numpy as np
import os

# Define the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Pretrained CLIP model name
clip_model_name = "openai/clip-vit-base-patch32"

# Load the processor and vision model
processor = CLIPProcessor.from_pretrained(clip_model_name)
vision_model = CLIPVisionModel.from_pretrained(clip_model_name).to(device)

# Optional projection layer
class EmbeddingProjector(nn.Module):
    def __init__(self, input_dim=768, output_dim=512):
        super().__init__()
        self.project = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.project(x)

projector = EmbeddingProjector().to(device)

def extract_and_save_embeddings(npy_files, output_file):
    """
    Extract embeddings from .npy files and save to a file.

    Args:
        npy_files (list): List of paths to .npy files containing image tensors.
        output_file (str): Path to save the embedding space.
    """
    embeddings_list = []

    for npy_file in npy_files:
        print(f"Processing file: {npy_file}")

        # Load .npy file (assumed shape: CxHxW or BxCxHxW)
        image_array = np.load(npy_file)  # NumPy array
        if image_array.ndim == 3:  # Single image
            image_tensor = torch.tensor(image_array).unsqueeze(0)  # Add batch dimension
        elif image_array.ndim == 4:  # Batch of images
            image_tensor = torch.tensor(image_array)
        else:
            raise ValueError("Unsupported .npy format: must be 3D (CxHxW) or 4D (BxCxHxW)")

        # Ensure proper shape and data type
        image_tensor = image_tensor.float().to(device)  # Convert to float and move to device

        # Preprocess and feed to CLIP model
        with torch.no_grad():
            inputs = {'pixel_values': image_tensor}
            vision_outputs = vision_model(**inputs)
            cls_embedding = vision_outputs.last_hidden_state[:, 0, :]  # CLS token
            projected_embedding = projector(cls_embedding)  # Optional projection
            embeddings_list.append(projected_embedding.cpu())

    # Combine and save embeddings
    embeddings_tensor = torch.cat(embeddings_list, dim=0)
    torch.save(embeddings_tensor, output_file)
    print(f"Embeddings saved to {output_file}")


# Example usage
if __name__ == "__main__":
    # Directory containing .npy files
    npy_dir = "/home/yyang239/portable_options/resources/dog_images"
    output_file = "clip_embeddings.pt"

    # List all .npy files
    npy_files = [os.path.join(npy_dir, f) for f in os.listdir(npy_dir) if f.endswith(".npy")]

    if not npy_files:
        raise FileNotFoundError("No .npy files found in the specified directory!")

    # Extract and save embeddings
    extract_and_save_embeddings(npy_files, output_file)

    # Verify saved embeddings
    saved_embeddings = torch.load(output_file)
    print(f"Loaded Embeddings Shape: {saved_embeddings.shape}")
