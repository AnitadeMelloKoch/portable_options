import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_distance(x):
    _x = x.detach()
    sim = torch.matmul(_x, _x.t())
    sim = torch.clamp(sim, max=1.0)
    dist = 2 - 2*sim
    dist += torch.eye(dist.shape[0]).to(dist.device)
    dist = dist.sqrt()
    
    return dist

class DistanceWeightedSampling(nn.Module):

    def __init__(self, batch_k=4, cutoff=0.5, nonzero_loss_cutoff=1.4, normalize=False, **kwargs):
        super(DistanceWeightedSampling, self).__init__()
        self.batch_k = batch_k
        self.cutoff = cutoff
        self.nonzero_loss_cutoff = nonzero_loss_cutoff
        self.normalize = normalize

    def forward(self, x):
        k = self.batch_k
        n, d = x.shape
        x_in = x
        x = F.normalize(x)
        distance = get_distance(x)
        distance = distance.clamp(min=self.cutoff)
        log_weights = ((2.0 - float(d)) * distance.log() - (float(d-3)/2)*torch.log(torch.clamp(1.0 - 0.25*(distance*distance), min=1e-8)))

        if self.normalize:
            log_weights = (log_weights - log_weights.min()) / (log_weights.max() - log_weights.min() + 1e-8)

        weights = torch.exp(log_weights - torch.max(log_weights))

        if x.device != weights.device:
            weights = weights.to(x.device)

        mask = torch.ones_like(weights)

        for i in range(0, n, k):
            mask[i:i+k, i:i+k] = 0

        mask_uniform_probs = mask.double()*(1.0/(n-k+1e-8))

        weights = weights*mask*((distance < self.nonzero_loss_cutoff).float()) + 1e-8
        weights_sum = torch.sum(weights, dim=1, keepdim=True)
        weights = weights/weights_sum

        anchor_indices = []
        positive_indices = []
        negative_indices = []

        np_weights = weights.cpu().numpy()

        for i in range(n):
            block_idx = i//k
            for j in range(block_idx*k, min((block_idx + 1)*k, n)):
                if j != i:
                    anchor_indices.append(i)
                    positive_indices.append(j)

        if weights_sum[i] != 0:
            negative_indices += np.random.choice(n, size=len(anchor_indices), p=np_weights[i]).tolist()
        else:
            negative_indices += np.random.choice(n, size=len(anchor_indices), p=mask_uniform_probs[i]).tolist()

        return anchor_indices, x_in[anchor_indices], x_in[positive_indices], x_in[negative_indices], x_in

