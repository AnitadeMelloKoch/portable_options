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


def choose_max_sum_qvals(q_vals):
    """
    choose actions by adding up the q-values of each action from all learners
    then choose the max-q value action
    args:
        q_vals: (num_learners, num_actions)
    return:
        action: an int
    """
    sumed_q_vals = np.sum(q_vals, axis=0)
    return np.argmax(sumed_q_vals)


def upper_confidence_bound(values, t, visitation_count, c=1):
    """
    an implementation of the upper confidence bound algorithm
    A_t = argmax_a [Q_t(a) + c * sqrt(2 * ln(t) / n(a))]
    args:
        values: (num_actions) the value estimates of each action
        t: the current timestep
        visitation_count: (num_actions) the number of times each action has been selected
        c: the constant used to balance exploration and exploitation
    """
    return np.argmax(values + c * np.sqrt(2 * np.log(t) / visitation_count))
