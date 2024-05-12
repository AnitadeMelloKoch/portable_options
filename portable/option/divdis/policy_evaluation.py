import numpy as np
from scipy.stats import wasserstein_distance_nd
from torch.nn.functional import kl_div
import torch

def get_wasserstain_distance(policy_a_samples, policy_b_samples):
    policy_a_samples = torch.softmax(policy_a_samples, dim=1)
    policy_b_samples = torch.softmax(policy_b_samples, dim=1)
    
    
    return wasserstein_distance_nd(
        policy_a_samples.numpy(),
        policy_b_samples.numpy()
    )

def get_kl_distance(policy_a_samples, policy_b_samples):
    policy_a_samples = torch.log_softmax(policy_a_samples, dim=1)
    policy_b_samples = torch.log_softmax(policy_b_samples, dim=1)
    return kl_div(
        policy_a_samples,
        policy_b_samples,
        log_target=True
    ).numpy()


def get_policy_similarity_metric(policy_a_samples, policy_b_samples, gamma=0.999, epsilon=1e-6, distribution=True, use_gpu=True):
    # Reference: https://doi.org/10.48550/arXiv.2101.05265
    if policy_a_samples.dim() == 1:
        policy_a_samples = policy_a_samples.unsqueeze(0)
    if policy_b_samples.dim() == 1:
        policy_b_samples = policy_b_samples.unsqueeze(0)
    
    policy_a_samples = torch.softmax(policy_a_samples, dim=-1) # q values, get action probs by softmax
    policy_b_samples = torch.softmax(policy_b_samples, dim=-1)
    n_a, n_b = policy_a_samples.shape[0], policy_b_samples.shape[0]
    action_cost_matrix = torch.zeros(n_a, n_b)
    if use_gpu:
        policy_a_samples = policy_a_samples.cuda()
        policy_b_samples = policy_b_samples.cuda()
        action_cost_matrix = action_cost_matrix.cuda()
    
    if distribution:
        prob_a_expanded = policy_a_samples.unsqueeze(1)
        prob_b_expanded = policy_a_samples.unsqueeze(0)
        action_cost_matrix = torch.sum(torch.abs(prob_a_expanded - prob_b_expanded), dim=2) / 2.0 # total variation distance
        
    else:
        policy_a_actions = torch.argmax(policy_a_samples, dim=-1)
        policy_b_actions = torch.argmax(policy_b_samples, dim=-1)
        action_cost_matrix = calculate_action_cost_matrix(policy_a_actions, policy_b_actions)
    
    n, m = action_cost_matrix.shape
    d_metric = torch.zeros_like(action_cost_matrix)
    
    # def fixed_point_operator(d_metric):
    #     d_metric_new = torch.empty_like(d_metric)
    #     for i in range(n):
    #         for j in range(m):
    #             d_metric_new[i, j] = action_cost_matrix[i, j] + gamma * d_metric[min(i + 1, n - 1), min(j + 1, m - 1)]
    #     return d_metric_new

    while True:
        #d_metric_new = fixed_point_operator(d_metric)
        right_shift_i = torch.arange(n).clamp(max=n-2) + 1
        right_shift_j = torch.arange(m).clamp(max=m-2) + 1
        d_metric_new = action_cost_matrix + gamma * d_metric[right_shift_i[:, None], right_shift_j] # use matrix operations

        # Check for convergence
        if torch.sum(torch.abs(d_metric - d_metric_new)) < epsilon:
            break
        else:
            d_metric = d_metric_new

    return torch.mean(d_metric)

    


def calculate_action_cost_matrix(policy_a_actions, policy_b_actions):
    # Reference: https://doi.org/10.48550/arXiv.2101.05265
    policy_a_actions = policy_a_actions.unsqueeze(1)  # Adds a new axis, converting shape from [n] to [n, 1]
    policy_b_actions = policy_b_actions.unsqueeze(0)  # Adds a new axis, converting shape from [m] to [1, m]

    action_equality = torch.eq(policy_a_actions, policy_b_actions)
    return 1.0 - action_equality.float()