import torch
import torch.nn as nn
import torch.nn.functional as F

from portable.option.ensemble import DistanceWeightedSampling

class AttentionModule(nn.Module):
    def __init__(self, in_channels, attention_depth) -> None:
        super().__init__()
        self.cnn1 = nn.Conv2d(in_channels=in_channels,
                              out_channels=attention_depth,
                              kernel_size=(4,4),
                              bias=False,
                              padding='same')
        # torch.nn.init.normal_(self.cnn1.weight)
        
        
    def forward(self, x):
        x = self.cnn1(x)
        # x = self.cnn2(x)
        
        return x
class SmallEmbedding(nn.Module):

    def __init__(self, stack_size=4, embedding_size=64, attention_depth=32, num_attention_modules=8, batch_k=4, normalize=False):
        super(SmallEmbedding, self).__init__()
        self.num_attention_modules = num_attention_modules
        self.out_dim = embedding_size
        self.attention_depth = attention_depth
        self.stack_size = stack_size
        
        self.sampled = DistanceWeightedSampling(batch_k=batch_k, normalize=normalize)

        self.spacial_feature_extractor_layers = self.build_spacial_layers()

        self.attention_modules = nn.ModuleList([
            AttentionModule(self.attention_depth, self.attention_depth)
            for i in range(self.num_attention_modules)])

        self.global_feature_extractor_layers = self.build_global_layers()

        self.linear = nn.LazyLinear(self.out_dim)

    def build_spacial_layers(self):
        layers = []
        
        layers.append(nn.Conv2d(in_channels=self.stack_size, 
                                out_channels=32, 
                                kernel_size=(3,3), 
                                stride=(1,1),
                                padding='same'))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=32, 
                                out_channels=32, 
                                kernel_size=(3,3), 
                                stride=(1,1), 
                                padding="same"))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        
        layers.append(nn.Conv2d(in_channels=32, 
                                out_channels=64, 
                                kernel_size=(3,3), 
                                stride=(1,1),
                                padding='same'))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=64, 
                                out_channels=self.attention_depth, 
                                kernel_size=(3,3), 
                                stride=(1,1), 
                                padding="same"))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        
        return nn.ModuleList(layers)

    def build_global_layers(self):
        layers = []
        
        layers.append(nn.Conv2d(in_channels=self.attention_depth, 
                                out_channels=128, 
                                kernel_size=(3,3), 
                                stride=(1,1), 
                                padding="same"))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=128, 
                                out_channels=128, 
                                kernel_size=(3,3), 
                                stride=(1,1), 
                                padding="same"))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        
        return nn.ModuleList(layers)

    def spatial_feature_extractor(self, x):
        for layer in self.spacial_feature_extractor_layers:
            x = layer(x)

        return x

    def global_feature_extractor(self, x):
        for layer in self.global_feature_extractor_layers:
            x = layer(x)

        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = F.normalize(x)

        return x

    def forward(self, x, sampling=False, return_attention_mask=False):
        spacial_features = self.spatial_feature_extractor(x)
        attentions = [self.attention_modules[i](spacial_features) for i in range(self.num_attention_modules)]

        for i in range(self.num_attention_modules):
            N, D, H, W = attentions[i].size()
            attention = attentions[i].view(-1, H*W)
            attention_max, _ = attention.max(dim=1, keepdim=True)
            attention_min, _ = attention.min(dim=1, keepdim=True)
            attentions[i] = ((attention - attention_min)/(attention_max-attention_min+1e-8)).view(N, D, H, W)

        embedding = torch.cat([self.global_feature_extractor(attentions[i]*spacial_features).unsqueeze(1) for i in range(self.num_attention_modules)], 1)

        if sampling is True:
            embedding = torch.flatten(embedding, 1)
            return self.sampled(embedding) if not return_attention_mask else (self.sampled(embedding), attentions)
        else:
            return embedding if not return_attention_mask else (embedding, attentions)
        
    def forward_one_attention(self, x, attention_idx):
        spacial_features = self.spatial_feature_extractor(x)
        attention = self.attention_modules[attention_idx](spacial_features)

        N, D, H, W = attention.size()
        attention = attention.view(-1, H*W)
        attention_max, _ = attention.max(dim=1, keepdim=True)
        attention_min, _ = attention.min(dim=1, keepdim=True)
        attention = ((attention-attention_min)/(attention_max-attention_min+1e-8)).view(N,D,H,W)

        embedding = torch.cat([self.global_feature_extractor(attention*spacial_features).unsqueeze(1)])

        return embedding
    
    def compute_l1_loss(self, w):
        return torch.abs(w).sum()