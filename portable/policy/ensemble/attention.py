import torch
import torch.nn as nn
import torch.nn.functional as F

from portable.plot import plot_attention_diversity
from portable.policy.models.impala import ConvSequence


class ImpalaAttentionEmbedding(nn.Module):
    """
    an attention model that attempts to follow the Impala arch
    """
    def __init__(self, 
                 obs_space,
                 embedding_size=64,
                 num_attention_modules=8,
                 plot_dir=None,):
        super().__init__()
        self.out_dim = embedding_size
        self.num_attention_modules = num_attention_modules
        self.plot_dir = plot_dir
        c, h, w = obs_space.shape
        shape = (c, h, w)

        self.spatial_conv = ConvSequence(input_shape=shape, out_channels=16)

        self.attention_modules = nn.ModuleList(
            [
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, bias=False)
                for _ in range(self.num_attention_modules)
            ]
        )
        self.attention_norm = nn.BatchNorm2d(16, affine=False)

        self.global_conv = ConvSequence(input_shape=(16, None, None), out_channels=32)
        self.linear = nn.LazyLinear(self.out_dim)

    def compact_global_features(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(x)
        x = self.linear(x)
        x = F.normalize(x)
        return x

    def forward(self, x, return_attention_mask=False, plot=False):
        spatial_features = self.spatial_conv(x)  # (N, 16, 32, 32)

        attentions = [self.attention_modules[i](spatial_features) for i in range(self.num_attention_modules)]
        attentions = [self.attention_norm(attention) for attention in attentions]

        global_features = [self.global_conv(attentions[i] * spatial_features) for i in range(self.num_attention_modules)]
        if plot:
            plot_attention_diversity(global_features, self.num_attention_modules, save_dir=self.plot_dir)
        embedding = torch.cat([self.compact_global_features(f).unsqueeze(1) for f in global_features], dim=1)  # (N, num_modules, embedding_size)

        return embedding if not return_attention_mask else (embedding, attentions)


class AttentionEmbedding(nn.Module):

    def __init__(self, 
                embedding_size=64, 
                attention_depth=32, 
                num_attention_modules=8, 
                use_individual_spatial_feature=False,
                use_individual_global_feature=False,
                plot_dir=None):
        super(AttentionEmbedding, self).__init__()
        self.num_attention_modules = num_attention_modules
        self.out_dim = embedding_size
        self.attention_depth = attention_depth
        self.use_individual_spatial_feature = use_individual_spatial_feature
        self.use_individual_global_feature = use_individual_global_feature

        if not self.use_individual_spatial_feature:
            self.conv1 = nn.LazyConv2d(out_channels=self.attention_depth, kernel_size=3, stride=1)
            self.pool1 = nn.MaxPool2d(2)
        else:
            self.conv1 = nn.ModuleList(
                [
                    nn.LazyConv2d(out_channels=self.attention_depth, kernel_size=3, stride=1)
                    for _ in range(self.num_attention_modules)
                ]
            )
            self.pool1 = nn.ModuleList(
                [
                    nn.MaxPool2d(2)
                    for _ in range(self.num_attention_modules)
                ]
            )

        self.attention_modules = nn.ModuleList(
            [
                nn.Conv2d(in_channels=self.attention_depth, out_channels=self.attention_depth, kernel_size=1, bias=False) 
                for _ in range(self.num_attention_modules)
            ]
        )

        if not self.use_individual_global_feature:
            self.conv2 = nn.Conv2d(in_channels=self.attention_depth, out_channels=64, kernel_size=3, stride=2)
            self.pool2 = nn.MaxPool2d(2)
        else:
            self.conv2 = nn.ModuleList(
                [
                    nn.Conv2d(in_channels=self.attention_depth, out_channels=64, kernel_size=3, stride=2)
                    for _ in range(self.num_attention_modules)
                ]
            )
            self.pool2 = nn.ModuleList(
                [
                    nn.MaxPool2d(2)
                    for _ in range(self.num_attention_modules)
                ]
            )
        
        self.linear = nn.LazyLinear(self.out_dim)

        self.plot_dir = plot_dir

    def spatial_feature_extractor(self, x):
        if not self.use_individual_spatial_feature:
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
        else:
            x = [F.relu(self.conv1[i](x)) for i in range(self.num_attention_modules)]
            x = [self.pool1[i](x[i]) for i in range(self.num_attention_modules)]

        return x

    def global_feature_extractor(self, x):
        if not self.use_individual_global_feature:
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
        else:
            x = [F.relu(self.conv2[i](x[i])) for i in range(self.num_attention_modules)]
            x = [self.pool2[i](x[i]) for i in range(self.num_attention_modules)]
        return x

    def compact_global_features(self, x):
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = F.normalize(x)
        return x

    def forward(self, x, return_attention_mask=False, plot=False):
        spacial_features = self.spatial_feature_extractor(x)
        if not self.use_individual_spatial_feature:
            spacial_features = [spacial_features] * self.num_attention_modules
        attentions = [self.attention_modules[i](spacial_features[i]) for i in range(self.num_attention_modules)]

        # normalize attention to between [0, 1]
        for i in range(self.num_attention_modules):
            N, D, H, W = attentions[i].size()
            attention = attentions[i].view(-1, H*W)
            attention_max, _ = attention.max(dim=1, keepdim=True)
            attention_min, _ = attention.min(dim=1, keepdim=True)
            attentions[i] = ((attention - attention_min)/(attention_max-attention_min+1e-8)).view(N, D, H, W)

        if not self.use_individual_global_feature:
            global_features = [self.global_feature_extractor(attentions[i] * spacial_features[i]) for i in range(self.num_attention_modules)]
        else:
            global_features = self.global_feature_extractor([attentions[i] * spacial_features[i] for i in range(self.num_attention_modules)])

        # normalize global features to between [0, 1]
        for i in range(self.num_attention_modules):
            N, D, H, W = global_features[i].size()
            feat = global_features[i].view(-1, H*W)
            feat_max, _ = feat.max(dim=1, keepdim=True)
            feat_min, _ = feat.min(dim=1, keepdim=True)
            global_features[i] = ((feat - feat_min)/(feat_max-feat_min+1e-8)).view(N, D, H, W)
        if plot:
            plot_attention_diversity(global_features, self.num_attention_modules, save_dir=self.plot_dir)

        # embedding = torch.cat([self.compact_global_features(f).unsqueeze(0) for f in global_features], dim=0)  # (num_modules, N, embedding_size)

        return global_features if not return_attention_mask else (global_features, attentions)
