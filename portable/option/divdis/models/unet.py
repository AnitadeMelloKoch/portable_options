import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, num_classes, num_heads):
        super(UNet, self).__init__()
        
        # Encoder path (downsampling)
        self.encoder = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample by factor of 2 (128 -> 64)
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample by factor of 2 (64 -> 32)
        ) for _ in range(num_heads)])
        
        # Bottleneck (bridge between encoder and decoder)
        self.bottleneck = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ) for _ in range(num_heads)])
        
        # Decoder path (upsampling)
        self.decoder = nn.ModuleList([nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2),  # Upsample (32 -> 64)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2),  # Upsample (64 -> 128)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        ) for _ in range(num_heads)])
        
        # Final classification layer for each head
        self.classifier = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=num_classes, kernel_size=1),  # 1x1 conv to get class predictions
            nn.Flatten(),
            nn.LazyLinear(500),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.LazyLinear(100),
            nn.ReLU(),
            nn.LazyLinear(num_classes)
        ) for _ in range(num_heads)])
        
        self.num_heads = num_heads

    def forward(self, x, logits=False):
        pred = torch.zeros(x.shape[0], self.num_heads, self.classifier[0][-1].out_features).to(x.device)
        
        for idx in range(self.num_heads):
            # Encoder
            enc = self.encoder[idx](x)
            
            # Bottleneck
            bottleneck = self.bottleneck[idx](enc)
            
            # Decoder
            dec = self.decoder[idx](bottleneck)
            
            # Classification
            y = self.classifier[idx](dec)
            
            # Apply softmax if not logits
            if not logits:
                y = F.softmax(y, dim=-1)
                
            pred[:, idx, :] = y
            
        return pred
