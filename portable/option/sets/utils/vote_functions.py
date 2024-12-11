import numpy as np

VOTE_FUNCTION_NAMES = [
    "weighted_vote_low",
    "weighted_vote_high"
]

def get_vote_function(vote_function_name):
    if vote_function_name == "weighted_vote_low":
        return weighted_vote_low

    if vote_function_name == "weighted_vote_high":
        return weighted_vote_high

def weighted_vote_low(votes, vote_confidences, weights):
    votes_weighted = np.multiply(np.multiply(votes, vote_confidences), weights)
    if np.sum(votes_weighted) >= 0.4:
        return 1
    else:
        return 0

def weighted_vote_high(votes, vote_confidences, weights):
    votes_weighted = np.multiply(np.multiply(votes, vote_confidences), weights)
    if np.sum(votes_weighted) >= 2.5:
        return 1
    else:
        return 0