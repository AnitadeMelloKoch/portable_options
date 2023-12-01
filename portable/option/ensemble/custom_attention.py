import torch
import numpy as np 
import torch.nn as nn

import matplotlib.pyplot as plt
import gin

class AttentionLayer(nn.Module):
    # this layer learns a feature set
    def __init__(self,
                 embedding_size):
        super().__init__()
        
        self.attention_mask = nn.Parameter(
            # torch.randn(1, num_features, 1),
            torch.randn(1, embedding_size),
            requires_grad=True
        )
        # self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        mask = self.mask()
        x = x*mask
        
        return x
    
    def mask(self):
        # return self.softmax(self.attention_mask)
        return self.sigmoid(self.attention_mask)
    

class FactoredAttentionLayer(nn.Module):
    def __init__(self,
                 num_features):
        super().__init__()
        
        self.num_features = num_features
        self.attention_mask = nn.Parameter(
            torch.randn(1, num_features, 1),
            requires_grad=True
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = x*self.mask()
        
        return x
    
    def mask(self):
        return self.sigmoid(self.attention_mask)


class ClassificationHead(nn.Module):
    def __init__(self,
                 num_classes,
                 input_dim):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(),
            nn.BatchNorm1d(input_dim//2),
            nn.Linear(input_dim//2, input_dim//2),
            nn.ReLU(),
            nn.BatchNorm1d(input_dim//2),
            nn.Linear(input_dim//2, input_dim//4),
            nn.ReLU(),
            nn.BatchNorm1d(input_dim//4),
            nn.Linear(input_dim//4, input_dim//8),
            nn.ReLU(),
            nn.BatchNorm1d(input_dim//8),
            nn.Linear(input_dim//8, num_classes),
            nn.Softmax(-1)
        )
        
    
    def forward(self, x):
        x = self.network(x)
        
        return x

class AttentionSetII(nn.Module):
    # single ensemble member
    def __init__(self,
                 embedding_size,
                 num_classes):
        super().__init__()
        
        self.attention = AttentionLayer(embedding_size=embedding_size)
        self.classification = ClassificationHead(num_classes,embedding_size)
        
    def forward(self, x):
        x = self.attention(x)
        x = self.classification(x)
        
        return x

class AttentionEnsembleII(nn.Module):
    def __init__(self,
                 num_attention_heads,
                 embedding_size,
                 num_classes):
        super().__init__()
        
        self.attentions = nn.ModuleList(
            AttentionSetII(embedding_size=embedding_size,
                           num_classes=num_classes) for _ in range(num_attention_heads)
        )
        
        self.num_attention_heads = num_attention_heads
        self.embedding_size = embedding_size
        self.num_classes = num_classes
    
    def forward(self, x, concat_results=False):
        output = []
        
        for attention in self.attentions:
            if concat_results:
                att_out = attention(x).unsqueeze(1)
            else:
                att_out = attention(x)
            output.append(att_out)
        if concat_results:
            output = torch.cat(output, 1)
        
        return output

    def get_attention_masks(self):
        masks = []
        for attention in self.attentions:
            masks.append(attention.attention.mask())
        
        return masks

class PrintSize(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x


@gin.configurable
class AutoEncoder(nn.Module):
    def __init__(self,
                 num_input_channels,
                 feature_size,
                 image_height=84,
                 image_width=84):
        super().__init__()
        
        self.height_mult = image_height//4
        self.width_mult = image_width//4
        
        # if image_height == 200:
        #     self.height_mult = 38
        # if image_width == 200:
        #     self.width_mult = 38
        
        self.feature_size = feature_size
        self.num_input_channels = num_input_channels
                
        self.encoder = nn.Sequential(
            nn.Conv2d(num_input_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*self.height_mult*self.width_mult, feature_size),
        )
        
        self.decoder_linear = nn.Sequential(
            nn.Linear(feature_size, 64*self.height_mult*self.width_mult),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, padding=1, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32,16,3,padding=1,stride=2,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,16,3,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, num_input_channels, 3, padding=1),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder_linear(x)
        x = x.view(-1, 64, self.height_mult, self.width_mult)
        x = self.decoder(x)

        return x
    
    def feature_extractor(self, x):
        with torch.no_grad():
            x = self.encoder(x)
        
        return x
    
    def masked_image(self, x, mask):
        with torch.no_grad():
            x = self.encoder(x)
            x = mask*x
            x = self.decoder_linear(x)
            x = x.view(-1, 64, self.height_mult, self.width_mult)
            x = self.decoder(x)
        
        return x

class MockAutoEncoder(AutoEncoder):
    def __init__(self, 
                 num_input_channels=3, 
                 feature_size=10, 
                 image_height=84, 
                 image_width=84):
        super().__init__(num_input_channels, 
                         feature_size, 
                         image_height, 
                         image_width)
        
        self.encoder = None
        self.decoder_linear = None
        self.decoder = None
    
    def forward(self, x):
        return x
    
    def feature_extractor(self, x):
        return x
    
    def masked_image(self, x, mask):
        return x*mask

def encoder_loss(x, y):
    loss = torch.mean(torch.abs(torch.sum(x, dim=(1,2,3))-torch.sum(y, dim=(1,2,3))))
    return loss

def divergence_loss(masks, attention_idx):
    feature_size = masks[0].shape[1]
    attention_num = len(masks)
    
    feats1 = masks[attention_idx].squeeze().unsqueeze(0)
    feats1 = feats1.repeat(attention_num, 1)
    feats2 = torch.cat(masks).detach().squeeze()
    square_diffs = torch.square(feats1-feats2)
    summed_diff = torch.sum(square_diffs, dim=1)+1e-8
    dists = summed_diff**(1/2)
    # scale dists for importance scaling    
    loss = torch.mean(dists)
    loss = torch.clamp((feature_size**(1/2)) - loss, 0)
    
    return loss

def l1_loss(masks, attention_idx):
    return torch.norm(masks[attention_idx], 1)

def plot_attentioned_state(state, mask, save_dir):
    fig, ax = plt.subplots()
    sorted_indices = np.argsort(-mask)
    for idx in sorted_indices:
        ax.imshow(state[idx], alpha=float(mask[idx]))
    
    fig.savefig(save_dir)
    plt.close(fig)


