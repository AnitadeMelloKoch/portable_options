import torch
import numpy as np 
import torch.nn as nn

class AttentionLayer(nn.Module):
    # this layer learns an attention mask over
    # a set of features
    def __init__(self,
                 num_features):
        super().__init__()
        
        self.attention_mask = nn.Parameter(
            torch.randn(1, num_features, 1,1),
            requires_grad=True
        )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        mask = self.mask()
        x = x*mask
        
        return x
    
    def mask(self):
        return self.softmax(self.attention_mask)

class MLPLayers(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(18,9)
        self.linear2 = nn.Linear(9,9)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        
        return x

class CNN(nn.Module):
    def __init__(self,
                 num_input_features):
        super().__init__()
        self.cnn1 = nn.Conv2d(num_input_features, 8, 2, stride=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.cnn2 = nn.Conv2d(8, 16, 2, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        x = self.cnn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = self.relu(x)
        x = self.maxpool2(x)

        x = self.flatten(x)
        
        return x
        
class ClassificationHead(nn.Module):
    def __init__(self,
                 num_classes):
        super().__init__()
        self.linear1 = nn.Linear(400, 400)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(400, num_classes)
        self.softmax = nn.Softmax(-1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x =  self.softmax(x)
        
        return x

class AttentionSetII(nn.Module):
    # single ensemble member
    def __init__(self,
                 num_features,
                 num_classes):
        super().__init__()
        
        self.attention = AttentionLayer(num_features=num_features)
        self.cnn = CNN(num_features)
        # self.mlp_layers = MLPLayers()
        self.classification = ClassificationHead(num_classes)
        
    def forward(self, x):
        x = self.attention(x)
        x = self.cnn(x)
        # x = self.mlp_layers(x)
        x = self.classification(x)
        
        return x

class AttentionEnsembleII(nn.Module):
    def __init__(self,
                 num_attention_heads,
                 num_features,
                 num_classes):
        super().__init__()
        
        self.attentions = nn.ModuleList(
            AttentionSetII(num_features=num_features,
                           num_classes=num_classes) for _ in range(num_attention_heads)
        )
        
        self.num_attention_heads = num_attention_heads
        self.num_features = num_features
        self.num_classes = num_classes
    
    def forward(self, x):
        output = []
        for attention in self.attentions:
            output.append(attention(x))
        
        return output

    def get_attention_masks(self):
        masks = []
        for attention in self.attentions:
            masks.append(attention.attention.mask())
        
        return masks

def divergence_loss(masks, attention_idx):
    attention_num = len(masks)
    
    feats1 = masks[attention_idx].squeeze().unsqueeze(0)
    feats1 = feats1.repeat(attention_num, 1)
    feats2 = torch.cat(masks).detach().squeeze()
    square_diffs = torch.square(feats1-feats2)
    summed_diff = torch.sum(square_diffs, dim=1)+1e-8
    dists = summed_diff**(1/2)
    # scale dists for importance scaling    
    loss = torch.sum(dists)
    loss = torch.clamp(attention_num - loss, 0)/attention_num
    
    return loss

def l1_loss(masks, attention_idx):
    return torch.norm(masks[attention_idx], 1)
