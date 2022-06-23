"""
aggregate the policy output of the ensemble into a single action
"""
import numpy as np


def choose_most_popular(actions):
    """
    given a list of actions, choose the most popular one
    """
    counts = np.bincount(actions)
    return np.argmax(counts)


def choose_leader(actions, leader=None):
    """
    choose a `leader` and all learners in the ensemble will listen to that 
    leader for action selection

    the hope is eventually all learners will converge to the same action selection policy

    args:
        leader: an int indicating which learner is the leader. The calling function is
            responsible for setting this value.
            The leader in evey episode should be the same.
    """
    if leader is None:
        return np.random.choice(actions)
    else:
        return actions[leader]