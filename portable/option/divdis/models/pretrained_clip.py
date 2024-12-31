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
        print(f"Image shape: {image_array.shape}")

        # Use the CLIP processor to preprocess the image
        inputs = processor(images=image_array, return_tensors="pt", padding=True).to(device)
        
        # Extract embeddings
        with torch.no_grad():
            vision_outputs = vision_model(**inputs)
            cls_embedding = vision_outputs.last_hidden_state[:, 0, :]  # CLS token
            embeddings_list.append(cls_embedding.cpu())

    # Combine and save embeddings
    embeddings_tensor = torch.cat(embeddings_list, dim=0)
    print(f"Embeddings shape before saving: {embeddings_tensor.shape}")
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
