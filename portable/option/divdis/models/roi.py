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
        print(f"Input to ROIClassifier: {x.shape}")
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        print(f"After conv1: {x.shape}")
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        print(f"After conv2: {x.shape}")
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        print(f"After conv3: {x.shape}")
        return x

class RoIPool(nn.Module):
    def __init__(self, num_classes, num_heads):
        super(RoIPool, self).__init__()
        in_channels = 4

        self.num_heads = num_heads
        self.num_classes = num_classes
        
        self.layer1 = nn.ModuleList([ROIClassifier(in_channels, num_classes) for _ in range(num_heads)])
        self.layer2 = nn.Linear(64 * 2 * 2, num_classes)  

    def forward(self, x, logits=False):
        batch_size = x.size(0)
        channels, height, width = x.size(1), x.size(2), x.size(3)

        pooled_features = []

        # Assuming rois cover the entire feature map for simplicity
        for idx in range(batch_size):
            roi_feature_map = x[idx:idx+1, :, :height, :width]
            pooled_feature = F.adaptive_max_pool2d(roi_feature_map, output_size=(2, 2))
            pooled_features.append(pooled_feature)

        pooled_features = torch.cat(pooled_features, dim=0)
        head_outputs = [head(pooled_features) for head in self.layer1]

        # Stack outputs and average them
        combined_output = torch.stack(head_outputs).mean(dim=0)

        if logits:
            final_output = self.layer2(combined_output)
            return final_output
        else:
            return combined_output

# Example usage
if __name__ == "__main__":
    # Create dummy feature maps (batch_size=64, channels=4, height=10, width=10)
    feature_maps = torch.randn(64, 4, 10, 10)

    roi_pool = RoIPool(num_classes=10, num_heads=3)
    pooled_features = roi_pool(feature_maps, logits=False)

    print(f"Output shape: {pooled_features.shape}")
