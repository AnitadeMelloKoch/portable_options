"""Implementation of a Rainbow agent with bootstrap for skills"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from bonus_based_exploration.noisy_networks import noisy_dqn_agent

from dopamine.agents.rainbow import rainbow_agent as base_rainbow_agent
from dopamine.replay_memory import prioritized_replay_buffer
from dopamine.replay_memory.circular_replay_buffer import ReplayElement
import gin
import numpy as np
# import tensorflow.compat.v1 as tf
import tensorflow as tf
import random

tf.compat.v1.disable_v2_behavior()
# from tensorflow.contrib import layers as contrib_layers
import tf_slim as contrib_slim

slim = contrib_slim

@gin.configurable
class NoisyRainbowAgentForSkills(base_rainbow_agent.RainbowAgent):
    """A Rainbow agent with noisy networks."""
    def __init__(self,
                 sess,
                 num_actions,
                 available_actions,
                 num_atoms=51,
                 vmax=10.,
                 gamma=0.99,
                 update_horizon=1,
                 min_replay_history=20000,
                 update_period=4,
                 target_update_period=8000,
                 epsilon_fn=lambda w, x, y, z: 0,
                 epsilon_decay_period=250000,
                 replay_scheme='prioritized',
                 tf_device='/cpu:*',
                 use_staging=True,
                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.00025, epsilon=0.0003125),
                 summary_writer=None,
                 summary_writing_frequency=500,
                 noise_distribution='independent'):
        """Initializes the agent and constructs the components of its graph.

    Args:
      sess: `tf.Session`, for executing ops.
      num_actions: int, number of actions the agent can take at any state.
      num_atoms: int, the number of buckets of the value function distribution.
      vmax: float, the value distribution support is [-vmax, vmax].
      gamma: float, discount factor with the usual RL meaning.
      update_horizon: int, horizon at which updates are performed, the 'n' in
        n-step update.
      min_replay_history: int, number of transitions that should be experienced
        before the agent begins training its value function.
      update_period: int, period between DQN updates.
      target_update_period: int, update period for the target network.
      epsilon_fn: function expecting 4 parameters:
        (decay_period, step, warmup_steps, epsilon). This function should return
        the epsilon value used for exploration during training.
      epsilon_decay_period: int, length of the epsilon decay schedule.
      replay_scheme: str, 'prioritized' or 'uniform', the sampling scheme of the
        replay memory.
      tf_device: str, Tensorflow device on which the agent's graph is executed.
      use_staging: bool, when True use a staging area to prefetch the next
        training batch, speeding training up by about 30%.
      optimizer: tf.train.Optimizer, for training the value function.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
      noise_distribution: string, distribution used to sample noise, must be
        `factorised` or `independent`.
    """
        self.noise_distribution = noise_distribution
        self.available_actions = available_actions
        self.mask = None
        self.mask_shape = (num_actions, )
        super(NoisyRainbowAgentForSkills,
              self).__init__(sess=sess,
                             num_actions=num_actions,
                             num_atoms=num_atoms,
                             vmax=vmax,
                             gamma=gamma,
                             update_horizon=update_horizon,
                             min_replay_history=min_replay_history,
                             update_period=update_period,
                             target_update_period=target_update_period,
                             epsilon_fn=epsilon_fn,
                             epsilon_decay_period=epsilon_decay_period,
                             replay_scheme=replay_scheme,
                             tf_device=tf_device,
                             use_staging=use_staging,
                             optimizer=optimizer,
                             summary_writer=summary_writer,
                             summary_writing_frequency=summary_writing_frequency)

    def _network_template(self, state):
        """Builds the convolutional network used to compute the agent's Q-values.

    Args:
      state: `tf.Tensor`, contains the agent's current state.

    Returns:
      net: _network_type object containing the tensors output by the network.
    """
        weights_initializer = slim.variance_scaling_initializer(factor=1.0 / np.sqrt(3.0),
                                                                mode='FAN_IN',
                                                                uniform=True)

        net = tf.cast(state, tf.float32)
        net = tf.div(net, 255.)
        net = slim.conv2d(net, 32, [8, 8], stride=4, weights_initializer=weights_initializer)
        net = slim.conv2d(net, 64, [4, 4], stride=2, weights_initializer=weights_initializer)
        net = slim.conv2d(net, 64, [3, 3], stride=1, weights_initializer=weights_initializer)
        net = slim.flatten(net)
        net = noisy_dqn_agent.fully_connected(net, 512, scope='fully_connected', distribution=self.noise_distribution)
        net = noisy_dqn_agent.fully_connected(net,
                                              self.num_actions * self._num_atoms,
                                              activation_fn=None,
                                              distribution=self.noise_distribution,
                                              scope='fully_connected_1')

        logits = tf.reshape(net, [-1, self.num_actions, self._num_atoms])
        probabilities = slim.softmax(logits)
        q_values = tf.reduce_sum(self._support * probabilities, axis=2)
        return self._get_network_type()(q_values, logits, probabilities)

    def _select_action(self):
        """Select an action from the set of available actions.
        Chooses an action randomly with probability self._calculate_epsilon(), and
        otherwise acts greedily according to the current Q-value estimates.
        Returns:
           int, the selected action.
        """
        if self.eval_mode:
            epsilon = self.epsilon_eval
        else:
            epsilon = self.epsilon_fn(self.epsilon_decay_period, self.training_steps,
                                      self.min_replay_history, self.epsilon_train)
        mask = self.available_actions()
        self.mask = mask

        # print(mask)

        if random.random() <= epsilon:
            # Choose a random action with probability epsilon.
            p = mask / mask.sum()
            return np.random.choice(np.arange(self.num_actions), p=p)
        else:
            # Choose the action with highest Q-value at the current state.
            tensor_mask = tf.convert_to_tensor(mask, dtype=tf.bool)
            tensor_mask = tf.expand_dims(tensor_mask, axis=0)
            tensor_mask = tf.tile(tensor_mask, tf.convert_to_tensor([self.state.shape[0], 1]))
            inf = tf.tile(tf.convert_to_tensor([[-np.inf]]),
                          tf.convert_to_tensor([self.state.shape[0],
                                                len(mask)]))
            qs = tf.where(tensor_mask, self._net_outputs.q_values, inf)
            qs_masked_max = tf.argmax(qs, axis=1)[0]
            # print(self._sess.run(self._net_outputs.q_values, {self.state_ph: self.state}))
            # print(self._sess.run(qs, {self.state_ph: self.state}))
            # print(self._sess.run(qs_masked_max, {self.state_ph: self.state}))
            return self._sess.run(qs_masked_max, {self.state_ph: self.state})

    def _store_transition(self, 
                          last_observation, 
                          action, 
                          reward, 
                          is_terminal, 
                          mask,
                          priority=None):
        """Stores a transition when in training mode.
        Executes a tf session and executes replay buffer ops in order to store the
        following tuple in the replay buffer (last_observation, action, reward,
        is_terminal, priority).
        Args:
          last_observation: Last observation, type determined via observation_type
            parameter in the replay_memory constructor.
          action: An integer, the action taken.
          reward: A float, the reward.
          is_terminal: Boolean indicating if the current state is a terminal state.
          priority: Float. Priority of sampling the transition. If None, the default
            priority will be used. If replay scheme is uniform, the default priority
            is 1. If the replay scheme is prioritized, the default priority is the
            maximum ever seen [Schaul et al., 2015].
        """
        if priority is None:
            if self._replay_scheme == 'uniform':
                priority = 1.
            else:
                priority = self._replay.memory.sum_tree.max_recorded_priority

        if not self.eval_mode:
            self._replay.add(last_observation, action, reward, is_terminal, mask, priority)

    def _build_replay_buffer(self, use_staging):
       """Creates the replay buffer used by the agent.
       Args:
         use_staging: bool, if True, uses a staging area to prefetch data for
           faster training.
       Returns:
         A `WrappedPrioritizedReplayBuffer` object.
       Raises:
         ValueError: if given an invalid replay scheme.
       """
       if self._replay_scheme not in ['uniform', 'prioritized']:
         raise ValueError('Invalid replay scheme: {}'.format(self._replay_scheme))
       # Both replay schemes use the same data structure, but the 'uniform' scheme
       # sets all priorities to the same value (which yields uniform sampling).
       return prioritized_replay_buffer.WrappedPrioritizedReplayBuffer(
           observation_shape=self.observation_shape,
           stack_size=self.stack_size,
           extra_storage_types=[ReplayElement("mask", self.mask_shape, np.bool)],
           use_staging=use_staging,
           update_horizon=self.update_horizon,
           gamma=self.gamma,
           observation_dtype=self.observation_dtype.as_numpy_dtype)

    def _build_target_distribution(self):
        """Builds the C51 target distribution as per Bellemare et al. (2017).
        First, we compute the support of the Bellman target, r + gamma Z'. Where Z'
        is the support of the next state distribution:
          * Evenly spaced in [-vmax, vmax] if the current state is nonterminal;
          * 0 otherwise (duplicated num_atoms times).
        Second, we compute the next-state probabilities, corresponding to the action
        with highest expected value.
        Finally we project the Bellman target (support + probabilities) onto the
        original support.
        Returns:
          target_distribution: tf.tensor, the target distribution from the replay.
        """
        batch_size = self._replay.batch_size

        # size of rewards: batch_size x 1
        rewards = self._replay.rewards[:, None]

        # size of tiled_support: batch_size x num_atoms
        tiled_support = tf.tile(self._support, [batch_size])
        tiled_support = tf.reshape(tiled_support, [batch_size, self._num_atoms])

        # size of target_support: batch_size x num_atoms

        is_terminal_multiplier = 1. - tf.cast(self._replay.terminals, tf.float32)
        # Incorporate terminal state to discount factor.
        # size of gamma_with_terminal: batch_size x 1
        gamma_with_terminal = self.cumulative_gamma * is_terminal_multiplier
        gamma_with_terminal = gamma_with_terminal[:, None]

        target_support = rewards + gamma_with_terminal * tiled_support

        # mask target q values
        mask = self._replay.transition["mask"]
        inf = tf.tile(
          tf.convert_to_tensor([[-np.inf]]),
          tf.convert_to_tensor([batch_size, self.num_actions])
        )
        next_qt = tf.where(mask, self._replay_next_target_net_outputs.q_values, inf)
        next_qt_argmax = tf.argmax(next_qt, axis=1)[:, None]

        batch_indices = tf.range(tf.cast(batch_size, tf.int64))[:, None]
        # size of next_qt_argmax: batch_size x 2
        batch_indexed_next_qt_argmax = tf.concat(
            [batch_indices, next_qt_argmax], axis=1)

        # size of next_probabilities: batch_size x num_atoms
        next_probabilities = tf.gather_nd(
            self._replay_next_target_net_outputs.probabilities,
            batch_indexed_next_qt_argmax)

        return base_rainbow_agent.project_distribution(target_support, next_probabilities,
                                    self._support)

    def step(self, reward, observation):
        """Records the most recent transition and returns the agent's next action.
        We store the observation of the last time step since we want to store it
        with the reward.
        Args:
          reward: float, the reward received from the agent's most recent action.
          observation: numpy array, the most recent observation.
        Returns:
          int, the selected action.
        """
        self._last_observation = self._observation
        self._record_observation(observation)

        if not self.eval_mode:
            self._store_transition(self._last_observation, self.action, reward, False, self.mask)
            self._train_step()

        self.action = self._select_action()

        return self.action

    def end_episode(self, reward):
        """Signals the end of the episode to the agent.
        We store the observation of the current time step, which is the last
        observation of the episode.
        Args:
          reward: float, the last reward from the environment.
        """
        if not self.eval_mode:
            self._store_transition(self._observation, self.action, reward, True, self.mask)


