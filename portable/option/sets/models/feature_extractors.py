import torch
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt 

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
        y = x[1]
        x = x[0]
        device = x.device
        num_batches, num_channels, _, _ = x.shape
        new_batch = np.zeros((num_batches, num_channels, 32, 24))
        x = x.cpu().numpy()
        for batch in range(num_batches):
            for channel in range(num_channels):
                img = Image.fromarray(x[batch, channel, :,:])
                new_batch[batch, channel, :, :] = np.asarray(img.resize((24,32), Image.BILINEAR))
            # plot_image(new_batch[batch], y[batch])
        new_batch = torch.from_numpy(new_batch).to(device).float()
        return new_batch.view(-1, 6, 768)
    
    # try with "ram info"

def plot_image(image, y):
    print(image.shape)
    fig, axes = plt.subplots(nrows=1, ncols=6)
    for idx, ax in enumerate(axes):
        ax.set_axis_off()
        ax.imshow(image[idx], cmap='gray')
    plt.show(block=False)
    input("continue: class {}".format(y))
    plt.close(fig)

class FactoredMinigridPositionFeatureExtractor(torch.nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.objects = ["agent",
                        "key",
                        "door",
                        "goal",
                        "box",
                        "ball"]
        self.device = device
    
    def forward(self, x):
        # print(x)
        num_batches = x.shape[0]
        new_batch = np.zeros((num_batches, 6, 3))
        for b_idx in range(num_batches):
            for o_idx, obj in enumerate(self.objects):
                new_batch[b_idx, o_idx] = x[b_idx][obj]
        
        new_batch = torch.from_numpy(new_batch).to(self.device).float()
        
        return new_batch
                
                
                

