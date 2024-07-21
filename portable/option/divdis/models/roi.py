import torch
import torch.nn as nn
import torch.nn.functional as F

class ROIClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ROIClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 2 * 2, 512)  # Adjust input features according to the flattened size
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

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
        in_channels = 4  # Adjust based on your input channels

        self.num_heads = num_heads
        self.num_classes = num_classes
        
        self.layer1 = nn.ModuleList([ROIClassifier(in_channels, num_classes) for _ in range(num_heads)])
        self.layer2 = nn.Linear(num_classes, num_classes)

    def forward(self, x, logits=False):
        """
        Args:
            x (torch.Tensor): Feature maps from the backbone network with shape (batch_size, channels, height, width).
            logits (bool): If True, apply the final linear layer (self.layer2). Otherwise, return the raw features.

        Returns:
            torch.Tensor: Final output tensor after RoI pooling and head processing.
        """
        batch_size = x.size(0)
        height, width = x.size(2), x.size(3)

        pooled_features = []

        rois = [[1, height, width] for _ in range(batch_size)]

        for roi in rois:
            r, h, w = roi
            roi_feature_map = x[:, :, r:h, r:w]  
            pooled_feature_map = F.adaptive_max_pool2d(roi_feature_map, output_size=(2, 2))
            pooled_features.append(pooled_feature_map)

        pooled_features = torch.cat(pooled_features, dim=0)

        head_outputs = [head(pooled_features) for head in self.layer1]

        combined_output = torch.stack(head_outputs).mean(dim=0)

        if logits:
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
