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


def upper_confidence_bound_with_gestation(values, t, visitation_count, gestation_period, c=1):
    """
    the same bandit as upper_confidence_bound
    but when t <= gestation_period, the agent will choose actions uniformly at random
    """
    if t <= gestation_period:
        return np.random.choice(len(values))
    else:
        return upper_confidence_bound(values, t, visitation_count, c)


def upper_confidence_bound_agent_57(mean_rewards, t, visitation_count, beta=1):
    """
    the (stationary) upper confidence bound algorithm from the agent 57 paper
    https://arxiv.org/pdf/2003.13350.pdf
    after k steps:
    for 0 <= k <= N - 1:
        A_t = k
    for N <= k <= K-1:
        A_t = argmax_a [miu_a + beta * sqrt(ln(k-1) / N(a))]
    """
    if t <= len(mean_rewards) - 1:
        return t
    else:
        return np.argmax(mean_rewards + beta * np.sqrt(np.log(t - 1) / visitation_count))


def upper_confidence_bound_with_window_size(mean_rewards, t, visitation_count, beta=1, epsilon=0.1):
    """
    the nonstationary upper confidence bound algorithm from the agent 57 paper
    https://arxiv.org/pdf/2003.13350.pdf
    This is the simplified sliding window UCB with epsilon-greedy exploration
    for 0 <= k <= N - 1:
        A_t = k
    for N <= k <= K-1 and U_k >= epsilon:
        A_t = argmax_a [miu_a_tau + beta * sqrt(1 / N(a, tau))]
    for N <= k <= K-1 and U_k < epsilon:
        A_t = random uniform
    """
    if t <= len(mean_rewards) - 1:
        return t
    else:
        u = np.random.uniform()
        if u >= epsilon:
            return np.argmax(mean_rewards + beta * np.sqrt(1 / visitation_count))
        else:
            return np.random.choice(len(mean_rewards))


weights = None


def exp3_bandit_algorithm(reward, num_arms, gamma=0.1):
    """
    an implementation of the exp3 bandit algorithm
    See: https://cseweb.ucsd.edu/~yfreund/papers/bandits.pdf
    args:
        gamma: a parameter in (0, 1] that controls exploration (I think) by mixing a uniform distribution with the current weights
    """
    # make sure weights are initialized
    global weights
    if weights is None:
        weights = np.ones(num_arms)
    # choose an action
    probs = (1 - gamma) * weights / np.sum(weights) + gamma / len(weights)
    a = np.random.choice(len(probs), p=probs)
    # update weights
    x_hat = np.zeros_like(weights)
    x_hat[a] = reward / probs[a]
    weights = weights * np.exp(gamma * x_hat / len(weights))
    # my own addition - keep weights from exploding
    if np.max(weights) > 100:
        weights = weights / np.max(weights)
    return a
