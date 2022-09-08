import numpy as np

VOTE_FUNCTION_NAMES = [
    "weighted_vote"
]

def get_vote_function(vote_function_name):
    if vote_function_name == "weighted_vote":
        return weighted_vote

def weighted_vote(votes, vote_confidences, weights):
    votes_weighted = np.multiply(np.multiply(votes, vote_confidences), weights)
    if np.sum(votes_weighted) >= 1.4:
        return 1
    else:
        return 0

