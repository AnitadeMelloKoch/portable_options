import torch
import numpy as np


n_modules = None  # number of modules, global variable that needs to be initialized once
every_tuple = None  # initialized in batched_L_divergence(), so that we don't repeatedly compute it
def L_divergence(feats):
    """
    feats is of shape (n_modules, n_features)
    """
    n_modules, _ = feats.shape
    feat_1 = feats.repeat(n_modules, 1)
    feat_2 = torch.repeat_interleave(feats, torch.tensor([n_modules]*n_modules, device=feats.device), dim=0)

    sums = torch.sum(torch.sub(feat_1, feat_2), dim=1).pow(2)
    sums = sums[sums.nonzero()].squeeze() # removes the elements that are 0 because we have subtracted a vector from itself
    loss = torch.clamp(1 - sums, min=0)

    return loss.mean()

def batched_L_divergence(batch_feats, weights):
    """
    batch_feats is of shape (batch_size, n_modules, n_features)
    """
    if batch_feats.shape[1] == 1:
        return 0  # no need to compute the loss if there is only one module

    global every_tuple
    if every_tuple is None:
        every_tuple = torch.combinations(torch.Tensor(range(batch_feats.shape[1])), 2).long()
        
    weights = torch.unsqueeze(weights, 1)
    weights = torch.unsqueeze(weights, 0)
    batch_feats = weights*batch_feats

    every_tuple_features = batch_feats[:, every_tuple, :]  # (batch_size, num_tuple, 2, dim)
    every_tuple_difference = every_tuple_features.diff(dim=2).squeeze(2)  # (batch_size, num_tuple, dim)
    loss = torch.clamp(1 - torch.sum(every_tuple_difference.pow(2), dim=-1), min=0)  # (batch_size, num_tuple)
    mean_loss = loss.sum(-1).mean()
    return mean_loss

def batched_criterion(feats, feat_class, weights):

    # x[:,:].view(1,-1)-x[:,:].view(-1,1)
    batch_size = feats.shape[0]
    num_modules = feats.shape[1]
    expanded_weights = torch.unsqueeze(weights, -1)
    expanded_weights = torch.unsqueeze(expanded_weights, 0)
    expanded_feats = expanded_weights*feats
    expanded_feats = torch.unsqueeze(expanded_feats, 1)
    expanded_feats = expanded_feats.repeat(1, batch_size, 1, 1)


    dist = torch.sum((expanded_feats-torch.transpose(expanded_feats, 0, 1)).pow(2), -1)
    expanded_classes = torch.unsqueeze(feat_class, 1).repeat(1, batch_size)

    loss = 0

    mask = expanded_classes == torch.transpose(expanded_classes, 0, 1)

    loss += torch.mean(dist[mask])
    loss += torch.mean(torch.clamp(1-dist[torch.logical_not(mask)], min=0))
    loss += batched_L_divergence(feats, weights)

    return loss

def L_metric(feat1, feat2, same_class=True):
    d = torch.sum((feat1 - feat2).pow(2).view((-1, feat1.size(-1))), 1)
    if same_class:
        return d.sum()/d.size(0)
    else:
        return torch.clamp(1 - d, min=0).sum() / d.size(0)


def loss_function(tensor, batch_k):
    batch_size = tensor.size(0)
    assert batch_size % batch_k == 0
    assert batch_k > 1
    loss_homo, loss_heter, loss_div = 0, 0, 0
    for i in range(batch_size):
        loss_div += L_divergence(tensor[i, ...])
    count_homo, count_heter = 0, 0
    for group_index in range(batch_size // batch_k):
        for i in range(batch_k):
            anchor = tensor[i+group_index*batch_k: 1+i+group_index*batch_k, ...]
            for j in range(i + 1, batch_k):
                index = j + group_index*batch_k
                loss_homo += L_metric(anchor, tensor[index: 1 + index, ...])
                count_homo += 1
            for j in range((group_index + 1)*batch_k, batch_size):
                loss_heter += L_metric(anchor, tensor[j:j + 1, ...])
                count_heter += 1
    
    return loss_div/batch_size, loss_homo/count_homo, loss_heter/count_heter

def criterion(anchors, positives, negatives):
    loss_homo = L_metric(anchors, positives)
    loss_heter = L_metric(anchors, negatives, False)
    loss_div = 0

    for i in range(anchors.shape[0]):
        loss_div += (L_divergence(anchors[i, ...]) + L_divergence(positives[i, ...]) + L_divergence(negatives[i, ...])) / 3
    
    return loss_div / anchors.shape[0], loss_homo, loss_heter