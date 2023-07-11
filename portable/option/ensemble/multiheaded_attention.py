import math 
import torch 
import copy
import matplotlib.pyplot as plt
import os
from einops import repeat, rearrange
from einops.layers.torch import Reduce, Rearrange
from portable.option.sets.models.feature_extractors import get_feature_extractor

class PrintLayer(torch.nn.Module):
    # print input. For debugging
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        print(x.shape)
        
        return x
    
class Embedding(torch.nn.Module):
    def __init__(self,
                 feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.cls_token = torch.nn.Parameter(torch.randn(1,1,feature_dim))
        
    def forward(self, x):
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=x.shape[0])
        # prepend cls token to input
        x = torch.cat([cls_tokens, x], dim=1)
        
        return x
        

class PositionEncoder(torch.nn.Module):
    def __init__(self,
                 feature_dim,
                 feature_num):
        super().__init__()
        self.positions = torch.nn.Parameter(torch.rand(feature_num + 1, feature_dim))
        
    def forward(self, x):
        x += self.positions
        return x
    
    
class MultiheadAttention(torch.nn.Module):
    def __init__(self,
                 attention_num,
                 feature_dim,
                 dropout_prob):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.attention_num = attention_num
        self.keys = torch.nn.Linear(feature_dim, feature_dim)
        self.queries = torch.nn.Linear(feature_dim, feature_dim)
        self.values = torch.nn.Linear(feature_dim, feature_dim)
        self.attn_dropout = torch.nn.Dropout(dropout_prob)
        self.projection = torch.nn.Linear(feature_dim, feature_dim)
    
    def forward(self, x, mask=None):
        # split keys, queries and values for attention num
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.attention_num)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.attention_num)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.attention_num)
        # sum over last axis
        scores = torch.einsum("bhqd, bhkd -> bhqk", queries, keys) # batch num, atn num, query len, key len
        if mask is not None:
            scores.masked_fill(-mask, float('-inf'))
        
        scaling = self.feature_dim**(1/2)
        att = torch.nn.functional.softmax(scores, dim=-1)/scaling
        att = self.attn_dropout(att)
        # sum over third axis
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        
        return out

class ResidualAdd(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network
    
    def forward(self, x, **kwargs):
        res = x
        x = self.network(x, **kwargs)
        x += res
        
        return x

class FeedForwardBlock(torch.nn.Sequential):
    def __init__(self,
                 feature_dim,                   # input embedding size
                 expansion,                     # factor by which we expand embedding for ffb 
                 dropout=0.1):
        super().__init__(
            torch.nn.Linear(feature_dim, expansion*feature_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(expansion*feature_dim, feature_dim)
        )
        
class TransformerEncoderBlock(torch.nn.Sequential):
    def __init__(self, 
                 feature_dim,
                 dropout_prob,
                 forward_expansion,
                 forward_dropout,
                 **kwargs):
        super().__init__(
            ResidualAdd(torch.nn.Sequential(
                torch.nn.LayerNorm(feature_dim),
                MultiheadAttention(feature_dim=feature_dim, dropout_prob=dropout_prob, **kwargs),
                torch.nn.Dropout(dropout_prob)
            )),ResidualAdd(torch.nn.Sequential(
                torch.nn.LayerNorm(feature_dim),
                FeedForwardBlock(
                    feature_dim,
                    forward_expansion,
                    forward_dropout
                ),
                torch.nn.Dropout(dropout_prob)
            ))
        )
    

class TransformerEncoder(torch.nn.Sequential):
    def __init__(self, depth, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])
    

class Classifier(torch.nn.Module):
    def __init__(self, 
                 feature_num,
                 feature_dim,
                 output_dim) -> None:
        super().__init__()
        
        self.linear1 = torch.nn.Linear(feature_dim*feature_num,
                                       1024)
        self.linear2 = torch.nn.Linear(1024,
                                       output_dim)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)
        
        return x

class ClassificationHead(torch.nn.Sequential):
    def __init__(self, feature_dim, n_classes):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            torch.nn.LayerNorm(feature_dim),
            torch.nn.Linear(feature_dim, feature_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(feature_dim, n_classes),
            torch.nn.Softmax(dim=-1)
        )

class PatchEmbedding(torch.nn.Module):
    def __init__(self, in_channels, patch_size, feature_dim, img_size):
        super().__init__()
        # self.patch_size = patch_size
        # self.projection = torch.nn.Sequential(
        #     torch.nn.Conv2d(in_channels,
        #                     feature_dim,
        #                     kernel_size=patch_size,
        #                     stride=patch_size),
        #     Rearrange('b e (h) (w) -> b (h w) e')
        # )
        self.feature_extractor = get_feature_extractor("factored_minigrid_images", {})
        self.flatten = torch.nn.Flatten(2)
        self.projection = torch.nn.Linear(img_size, feature_dim)
        
        self.cls_token = torch.nn.Parameter(torch.randn(1,1,feature_dim))
        self.positions = torch.nn.Parameter(
            torch.randn(in_channels, feature_dim)
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        x = self.projection(x)
        
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=x.shape[0])
        # x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions
        
        return x

class ViT(torch.nn.Sequential):
    def __init__(self,
                 in_channel,
                 patch_size,
                 feature_dim,
                 img_size,
                 depth,
                 n_classes,
                 **kwargs):
        super().__init__(
            PatchEmbedding(in_channel, patch_size, feature_dim, img_size),
            PrintLayer(),
            TransformerEncoder(depth, feature_dim=feature_dim, **kwargs),
            ClassificationHead(feature_dim, n_classes),
        )


def plot_scores(scores,
                save_path,
                epoch):
    
    for attn_idx in range(scores.shape[-1]):
        score = scores[0,:,:,attn_idx].detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(7.5,7.5))
        ax.matshow(score)
        for i in range(score.shape[0]):
            for j in range(score.shape[1]):
                ax.text(x=j, y=i, s="{:.2f}".format(score[i,j]),
                        va='center', ha='center')
        
        fig_dir = os.path.join(save_path, "attn_{}".format(attn_idx))
        os.makedirs(fig_dir, exist_ok=True)
        fig.savefig(fname=os.path.join(fig_dir, "epoch_{}.png".format(epoch)))
        plt.close(fig)


