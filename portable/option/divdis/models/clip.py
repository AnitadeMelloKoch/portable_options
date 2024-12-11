import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import CLIPModel

# Define the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Pretrained CLIP model name
clip_model_name = "openai/clip-vit-base-patch32"

# Define the preprocessing transformations manually
clip_image_transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Resize to 224x224, which is the input size for CLIP's vision model
    transforms.ToTensor(),           # Convert PIL Image or ndarray to tensor
    transforms.Normalize(            # Normalize using CLIP's ImageNet stats
        mean=[0.48145466, 0.4578275, 0.40821073], 
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])


class Clip(nn.Module):
    def __init__(self, num_classes, num_heads, embedding_dim=512):
        super().__init__()
        # Load only the vision part of the CLIP model
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).vision_model.to(device)

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

        self.num_heads = num_heads
        self.num_classes = num_classes
        self.full_model = nn.ModuleList([
            nn.Sequential(self.clip_model, classification_head)
            for classification_head in self.model
        ])
        # Linear layer to project the embedding dimension
        self.linear_layer = nn.Linear(768, embedding_dim).to(device)

    def forward(self, images):
        # Apply manual preprocessing to images
        if isinstance(images, torch.Tensor):
            # If images are already tensors, assume they are preprocessed
            pixel_values = images.to(device)
        else:
            # If images are PIL Images or ndarrays, apply preprocessing
            pixel_values = torch.stack([clip_image_transform(img) for img in images]).to(device)
        
        # Extract image features using the CLIP vision model
        with torch.no_grad():
            vision_outputs = self.clip_model(pixel_values=pixel_values, return_dict=False)

        print("vision_outputs pooler_output shape:", vision_outputs.pooler_output.shape)

        # Get the image embeddings and project to desired dimensions
        embeddings = vision_outputs.pooler_output  # Shape: [batch_size, 768]
        embeddings = self.linear_layer(embeddings)  # Transform to [batch_size, 512]
        print("embedding shape:", embeddings.shape)

        # Apply custom layers on the embeddings
        batch_size = embeddings.size(0)
        predictions = torch.zeros(batch_size, self.num_heads, self.num_classes, device=device)
        for idx in range(self.num_heads):
            predictions[:, idx, :] = self.model[idx](embeddings)

        # Apply softmax over the class dimension
        predictions = F.softmax(predictions, dim=-1)
        return predictions
