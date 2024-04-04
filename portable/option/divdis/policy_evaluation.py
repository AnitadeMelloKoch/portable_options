from scipy.stats import wasserstein_distance_nd
from torch.nn.functional import kl_div
import torch

def get_wasserstain_distance(policy_a_samples, policy_b_samples):
    return wasserstein_distance_nd(
        policy_a_samples,
        policy_b_samples
    )

def get_kl_distance(policy_a_samples, policy_b_samples):
    policy_a_samples = torch.log_softmax(policy_a_samples, dim=1)
    policy_b_samples = torch.log_softmax(policy_b_samples, dim=1)
    return kl_div(
        policy_a_samples,
        policy_b_samples,
        log_target=True
    )



