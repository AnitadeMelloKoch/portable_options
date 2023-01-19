# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implementation of a Rainbow agent with intrinsic rewards."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import random

from portable.agent.bonus_based_exploration.intrinsic_motivation import intrinsic_dqn_agent
from portable.agent.bonus_based_exploration.intrinsic_motivation import intrinsic_rewards
from portable.agent.bonus_based_exploration.replay_buffer import skill_prioritized_replay_buffer
from dopamine.agents.dqn import dqn_agent as base_dqn_agent
from dopamine.agents.rainbow import rainbow_agent as base_rainbow_agent
import gin
import tensorflow.compat.v1 as tf

@gin.configurable
class RNDRainbowAgentForSkills(base_rainbow_agent.RainbowAgent, intrinsic_dqn_agent.RNDDQNAgent):
    """A Rainbow agent paired with an intrinsic bonus derived from RND."""

    def __init__(self,
                 sess,
                 num_actions,
                 batch_size=32,
                 num_atoms=51,
                 vmax=10.,
                 gamma=0.99,
                 update_horizon=1,
                 min_replay_history=20000,
                 update_period=4,
                 target_update_period=8000,
                 epsilon_fn=intrinsic_dqn_agent.linearly_decaying_epsilon,
                 epsilon_train=0.01,
                 epsilon_eval=0.001,
                 epsilon_decay_period=250000,
                 replay_scheme='prioritized',
                 tf_device='/cpu:*',
                 use_staging=True,
                 optimizer=tf.train.AdamOptimizer(learning_rate=0.0000625, epsilon=0.00015),
                 summary_writer=None,
                 summary_writing_frequency=500,
                 clip_reward=False):
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
      epsilon_train: float, the value to which the agent's epsilon is eventually
        decayed during training.
      epsilon_eval: float, epsilon used when evaluating the agent.
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
      clip_reward: bool, whether or not clip the mixture of rewards.
    """
        self._clip_reward = clip_reward
        self.mask_shape = (num_actions, )
        self.intrinsic_model = intrinsic_rewards.RNDIntrinsicReward(sess=sess,
                                                                    tf_device=tf_device,
                                                                    summary_writer=summary_writer)
        with tf.device(tf_device):
            batched_shape = (None, ) + base_dqn_agent.NATURE_DQN_OBSERVATION_SHAPE + (
                base_dqn_agent.NATURE_DQN_STACK_SIZE, )
            self.batch_ph = tf.placeholder(base_dqn_agent.NATURE_DQN_DTYPE,
                                           batched_shape,
                                           name='observations_ph')
            obs_shape = (None, ) + base_dqn_agent.NATURE_DQN_OBSERVATION_SHAPE + (1, )
            self.obs_batch_ph = tf.placeholder(tf.uint8, shape=obs_shape, name='obs_batch_ph')
            self.mask_ph = tf.placeholder(tf.bool, shape=(None,self.mask_shape[0]), name='mask_ph')
            self.inf_ph = tf.placeholder(tf.float32, shape=(None,num_actions), name='inf_ph')
        super(RNDRainbowAgentForSkills,
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
                             epsilon_train=epsilon_train,
                             epsilon_eval=epsilon_eval,
                             epsilon_decay_period=epsilon_decay_period,
                             replay_scheme=replay_scheme,
                             tf_device=tf_device,
                             use_staging=use_staging,
                             optimizer=optimizer,
                             summary_writer=summary_writer,
                             summary_writing_frequency=summary_writing_frequency)

        self._last_state = np.copy(self.state)

    def _build_networks(self):
        super()._build_networks()

        self._net_outputs = self.online_convnet(self.batch_ph)
        self.value_function = tf.reduce_max(self._net_outputs.q_values, axis=1)
        self.reward_function = (self.intrinsic_model.loss -
                                self.intrinsic_model.reward_mean) / self.intrinsic_model.reward_std

        self._qs_masked_max = tf.argmax(
          tf.where(self.mask_ph, self._net_outputs.q_values, self.inf_ph),
          axis=1
        )[0]

    def _add_intrinsic_reward(self, observation, extrinsic_reward):
        return intrinsic_dqn_agent.RNDDQNAgent._add_intrinsic_reward(self, observation,
                                                                     extrinsic_reward)

    def _get_intrinsic_reward(self, observation):
        return self.intrinsic_model.compute_intrinsic_reward(observation, 0, True)

    def _get_value_function(self, stacks):
        return self._sess.run(self.value_function, {self.batch_ph: stacks})

    def get_q_values(self):
        return self._sess.run(self._net_outputs.q_values, {self.batch_ph: self.state})

    def select_action(self, mask):
        return self._select_action(mask)

    def _select_action(self, mask):
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

        if random.random() <= epsilon:
            # Choose a random action with probability epsilon.
            p = mask / mask.sum()
            return np.random.choice(np.arange(self.num_actions), p=p)
        else:
            mask = np.expand_dims(mask, axis=0)
            mask = np.tile(mask, (self.state.shape[0], 1))
            inf = np.tile([[-np.inf]],
                          (self.state.shape[0], self.num_actions))

            return self._sess.run(self._qs_masked_max, {self.batch_ph: self.state,
                                                        self.mask_ph: mask,
                                                        self.inf_ph: inf})

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


    def _store_transition(self,
                          last_state,
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
            # state is batched with size 1 but replay buffer does not expect batch so squeeze 
            self._replay.add(np.squeeze(last_state), action, reward, is_terminal, mask, priority)

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
        return skill_prioritized_replay_buffer.WrappedPrioritizedSkillReplayBuffer(
            observation_shape=self.observation_shape,
            stack_size=self.stack_size,
            mask_shape=self.mask_shape,
            mask_dtype=np.bool,
            use_staging=use_staging,
            update_horizon=self.update_horizon,
            gamma=self.gamma,
            observation_dtype=self.observation_dtype.as_numpy_dtype)

    def step(self, reward, state, action, mask):
        """Records the most recent transition and returns the agent's next action.
        We store the observation of the last time step since we want to store it
        with the reward.
        Args:
          reward: float, the reward received from the agent's most recent action.
          observation: numpy array, the most recent observation.
        Returns:
          int, the selected action.
        """
        self._last_state = np.copy(self.state)
        self._record_observation(state)

        if not self.eval_mode:
            self._store_transition(self._last_state, action, reward, False, mask)
            self._train_step()


    def _record_observation(self, state):
        """Records an observation and update state.
        Extracts a frame from the observation vector and overwrites the oldest
        frame in the state buffer.
        Args:
          observation: numpy array, an observation from the environment.
        """
        self.state = state

    def begin_episode(self, trajectory, mask):
        """Returns the agent's first action for this episode.
        Args:
          observation: numpy array, the environment's initial observation.
        Returns:
          int, the selected action.
        """
        self._reset_state()
        self._record_observation(trajectory)

        if not self.eval_mode:
          self._train_step()

        self.action = self._select_action(mask)
        return self.action

    def end_episode(self, reward, action, mask):
        """Signals the end of the episode to the agent.
        We store the observation of the current time step, which is the last
        observation of the episode.
        Args:
          reward: float, the last reward from the environment.
        """

        if not self.eval_mode:
          self._store_transition(self.state, action, reward, True, mask)
