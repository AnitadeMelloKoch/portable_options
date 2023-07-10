import torch
from PIL import Image
import numpy as np

EXTRACTOR_TYPES = [
    "factored_minigrid_images",
    "factored_minigrid_positions",
]

def get_feature_extractor(generator_type, kwargs):
    assert generator_type in EXTRACTOR_TYPES
    if generator_type == "factored_minigrid_images":
        return FactoredMinigridImageFeatureExtractor(**kwargs)
    if generator_type == "factored_minigrid_positions":
        return FactoredMinigridPositionFeatureExtractor

class FactoredMinigridImageFeatureExtractor(torch.nn.Module):
    # obs of factored minigrid already has 
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        device = x.device
        num_batches, num_channels, _, _ = x.shape
        new_batch = np.zeros((num_batches, num_channels, 24, 24))
        x = x.cpu().numpy()
        for batch in range(num_batches):
            for channel in range(num_channels):
                img = Image.fromarray(x[batch, channel, :,:])
                new_batch[batch, channel, :, :] = np.asarray(img.resize((24,24), Image.BILINEAR))
        new_batch = torch.from_numpy(new_batch).to(device).float()
        return new_batch.view(-1, 6, 576)
    
    # try with "ram info"

class FactoredMinigridPositionFeatureExtractor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.objects = ["agent",
                        "key",
                        "door",
                        "goal",
                        "box",
                        "ball"]
    
    def forward(self, x):
        batch = np.zeros([])
        print(x)
        for obj in self.objects:
            print(obj)

