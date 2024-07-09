import torch
import torch.nn as nn
import torch.nn.functional as F

class RoIPool(nn.Module):
    """
    RoIPool layer for extracting fixed-size feature maps from input feature maps based on regions of interest (RoIs).
    """
    def __init__(self, num_classes, num_heads):
        """
        Args:
            num_classes (int): Number of output classes.
            num_heads (int): Number of parallel RoI heads.
        """
        super(RoIPool, self).__init__()

        self.num_heads = num_heads
        self.num_classes = num_classes
        
        self.layer1 = None
        self.layer2 = nn.Linear(num_classes, num_classes)

    def _create_head_layers(self, in_channels):
        """
        Creates the first layer as a list of parallel head models based on input channels.
        """
        return nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 2 * 2, 512),  # Adjust input features according to the flattened size
                nn.ReLU(),
                nn.Linear(512, self.num_classes)
            ) for _ in range(self.num_heads)
        ])

    def forward(self, x, logits=False):
        """
        Args:
            x (torch.Tensor): Feature maps from the backbone network with shape (batch_size, channels, height, width).
            logits (bool): If True, apply the final linear layer (self.layer2). Otherwise, return the raw features.

        Returns:
            torch.Tensor: Final output tensor after RoI pooling and head processing.
        """
        batch_size = x.size(0)
        in_channels = x.size(1)

        
        if self.layer1 is None:
            self.layer1 = self._create_head_layers(in_channels)

        pooled_features = []

        for i in range(batch_size):  # Iterate over the batch size of feature_maps
            # Example RoI coordinates (adjust according to your actual RoIs)
            
            rois = [[1, 2, 6, 6]] * batch_size

            for roi in rois:
                x1, y1, x2, y2 = roi

                roi_feature_map = x[i, :, y1:y2, x1:x2]  
                pooled_feature_map = F.adaptive_max_pool2d(roi_feature_map, output_size=(2, 2))
                pooled_features.append(pooled_feature_map)

        pooled_features = torch.stack(pooled_features, dim=0)  

        head_outputs = [head(pooled_features) for head in self.layer1]

        combined_output = torch.stack(head_outputs).mean(dim=0)  

        if logits:
            # Apply layer2 to the combined output
            final_output = self.layer2(combined_output)
            return final_output
        else:
            return combined_output

# Example usage
if __name__ == "__main__":
    # Create dummy feature maps (batch_size=2, channels=256, height=10, width=10)
    feature_maps = torch.randn(2, 256, 10, 10)
    
    roi_pool = RoIPool(num_classes=10, num_heads=3)
    pooled_features = roi_pool(feature_maps, logits=False)

    print(pooled_features.shape)  # Should print (2, 10)
