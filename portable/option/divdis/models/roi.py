import torch
import torch.nn as nn
import torch.nn.functional as F

class PrintLayer(nn.Module):
    # Print input shape for debugging
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        print(x.shape)
        return x

import torch
import torch.nn as nn

class RoI(nn.Module):
    def __init__(self, output_size=(10, 10), spatial_scale=1.0/10):
        super(RoI, self).__init__()
        self.output_size = output_size  # (height, width)
        self.spatial_scale = spatial_scale  # scale factor to map RoIs to the feature map

    def forward(self, features = torch.randn(64, 512, 5, 5), rois = torch.tensor([[i, 100, 100, 200, 200] for i in range(64)])):
        """
        Args:
            features: Input feature maps from the backbone network, of shape (N, C, H, W)
            rois: List of RoIs, of shape (num_rois, 5), where each RoI is defined as 
                  (batch_index, x1, y1, x2, y2)

        Returns:
            Pooled features of shape (num_rois, C, output_size[0], output_size[1])
        """
        num_rois = rois.size(0)
        num_channels = features.size(1)
        output_height, output_width = self.output_size
        
        # Output tensor to store the pooled features
        pooled_features = torch.zeros((num_rois, num_channels, output_height, output_width),
                                      dtype=features.dtype, device=features.device)
        
        for i in range(num_rois):
            roi = rois[i]
            batch_index, x1, y1, x2, y2 = roi
            batch_index = int(batch_index)
            # Scale the RoI coordinates to match the feature map size
            x1 = int(x1 * self.spatial_scale)
            y1 = int(y1 * self.spatial_scale)
            x2 = int(x2 * self.spatial_scale)
            y2 = int(y2 * self.spatial_scale)
            
            roi_feature_map = features[batch_index, :, y1:y2, x1:x2]
            # print(roi_feature_map.shape)
            pooled = nn.functional.adaptive_max_pool2d(roi_feature_map, self.output_size)
            pooled_features[i] = pooled
        
        return pooled_features

# # Example usage:
# features = torch.randn(1, 512, 50, 50)  # Example feature map from a backbone
# rois = torch.tensor([[0, 10, 10, 30, 30], [0, 15, 20, 35, 40]])  # Example RoIs
# roi_pool = (output_size=(7, 7), spatial_scale=1.0/16)  # Output size (7, 7), with scaling factor

# pooled_features = roi_pool(features, rois)
# print(pooled_features.shape)  # Should print (2, 512, 7, 7)


class RoI_CNN(nn.Module):
    def __init__(self, num_classes, num_heads):
        super().__init__()
  
        
        self.model = nn.ModuleList([nn.Sequential(
            nn.LazyConv2d(out_channels=32, kernel_size=8, stride=4, padding=0, bias=False),
            
            nn.BatchNorm2d(32),

            ## Position 1 Accuracy stays 0.50
            nn.ReLU(),
            RoI(),
            
            
            nn.LazyConv2d(out_channels=64, kernel_size=4, stride=2, padding=0, bias=False),
            
            nn.BatchNorm2d(64),
            ## Position 2 Accuracy around: 0.47-0.50
            # RoI(),
            nn.ReLU(),
            
            nn.LazyConv2d(out_channels=64, kernel_size=3, stride=1, padding=0, bias=False),
            
            nn.BatchNorm2d(64),
            # Position 3 Accuracy around: 0.48-0.50
            # RoI(), 
            nn.ReLU(),
            
            nn.Flatten(),           
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.LazyLinear(num_classes)
            
        ) for _ in range(num_heads)])
        
        self.num_heads = num_heads
        self.num_classes = num_classes 

    def forward(self, x, logits=False):
        pred = torch.zeros(x.shape[0], self.num_heads, self.num_classes).to(x.device)
        for idx in range(self.num_heads):
            y = self.model[idx](x)
            if not logits:
                y = F.softmax(y, dim=-1)
            pred[:, idx, :] = y

        return pred


