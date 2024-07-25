import torch
import torch.nn as nn
import torch.nn.functional as F


class RoIPool(nn.Module):
    def __init__(self, num_classes, num_heads):
        super(RoIPool, self).__init__()

        self.model = nn.ModuleList([nn.Sequential(
            nn.LazyConv2d(out_channels=32, kernel_size=1, stride=4, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.LazyConv2d(out_channels=64, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.LazyConv2d(out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
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
        # Assuming rois cover the entire feature map for simplicity
        pooled_feature = F.adaptive_max_pool2d(x, output_size=(2, 2))
        for idx in range(self.num_heads):
            if logits:
                y = self.model[idx](pooled_feature)
            else:
                y = F.softmax(self.model[idx](pooled_feature), dim=-1)
            
            pred[:,idx,:] = y

        return pred

# Example usage
if __name__ == "__main__":
    # Create dummy feature maps (batch_size=64, channels=4, height=10, width=10)
    feature_maps = torch.randn(64, 4, 10, 10)

    roi_pool = RoIPool(num_classes=10, num_heads=3)
    pooled_features = roi_pool(feature_maps, logits=False)

    print(f"Output shape: {pooled_features.shape}")
