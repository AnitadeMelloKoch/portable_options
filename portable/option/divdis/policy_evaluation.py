from scipy.stats import wasserstein_distance_nd

def get_wasserstain_distance(policy_a_samples, policy_b_samples):
    return wasserstein_distance_nd(
        policy_a_samples,
        policy_b_samples
    )




