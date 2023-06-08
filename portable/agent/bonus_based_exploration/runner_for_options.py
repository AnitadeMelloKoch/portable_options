from cmath import inf
import os
import numpy as np
import sys

from dopamine.discrete_domains.run_experiment import TrainRunner
from dopamine.jax.agents.dqn import dqn_agent as jax_dqn_agent
from dopamine.discrete_domains import atari_lib
import gin
import tensorflow.compat.v1 as tf

from absl import logging

from portable.agent.bonus_based_exploration.helpers import env_wrapper, ram_data_replay_buffer
import matplotlib.pyplot as plt

# from bonus_based_exploration.helpers.video_generator import VideoGenerator

@gin.configurable
class TrainRunnerForOptions(TrainRunner):

    def __init__(self,
                 base_dir,
                 create_agent_fn,
                 create_environment_fn=atari_lib.create_atari_environment):
        super(TrainRunnerForOptions, self).__init__(base_dir=base_dir,
                                                   create_agent_fn=create_agent_fn,
                                                   create_environment_fn=create_environment_fn)

        if hasattr(self._environment, "environment"):
            self.env_wrapper = env_wrapper.MontezumaInfoWrapper(self._environment.environment)
            # self.info_buffer = ram_data_replay_buffer.MontezumaRevengeReplayBuffer(
            #     self._agent._replay.memory._replay_capacity
            # )
            # self.info_buffer.load(self._base_dir)
        elif hasattr(self._environment, "env"):
            self.env_wrapper = env_wrapper.MontezumaInfoWrapper(self._environment.env)
            # self.info_buffer = ram_data_replay_buffer.MontezumaRevengeReplayBuffer(
            #     self._agent._replay.memory._replay_capacity
            # )
            # self.info_buffer.load(self._base_dir)
        else:
            raise AttributeError(f"Could not find env attribute in {self._environment}")

        self._environment.game_over = False
        self.step_count = 0
        self.decision_count = 0
        # self.num_episodes = 0
        self.sum_returns = 0.
        self.episode = 0
        # self.checkpoint_number = 0
        # self.video_generator = VideoGenerator(base_dir + '/videos/')

        step_count_file = os.path.join(self._base_dir, 'checkpoints', 'step_count.npy')
        if os.path.exists(step_count_file):
            self.step_count = np.load(step_count_file)

        decision_count_file = os.path.join(self._base_dir, 'checkpoints', 'decision_count.npy')
        if os.path.exists(decision_count_file):
            self.decision_count = np.load(decision_count_file)
        # num_episodes_file = os.path.join(self._base_dir, 'checkpoints', 'num_episodes.npy')
        # if os.path.exists(num_episodes_file):
        #     self.num_episodes = np.load(num_episodes_file)
        # checkpoint_num_file = os.path.join(self._base_dir, 'checkpoints', 'checkpoint_num.npy')
        # if os.path.exists(checkpoint_num_file):
        #     self.checkpoint_number = int(np.load(checkpoint_num_file))


    def _checkpoint_experiment(self, iteration):
        np.save(os.path.join(self._base_dir, 'checkpoints', 'step_count.npy'), self.step_count)
        np.save(os.path.join(self._base_dir, 'checkpoints', 'decision_count.npy'), self.decision_count)
        # np.save(os.path.join(self._base_dir, 'checkpoints', 'num_episodes.npy'), self.num_episodes)
        # np.save(os.path.join(self._base_dir, 'checkpoints', 'checkpoint_num.npy'), self.checkpoint_number)

        return super()._checkpoint_experiment(iteration)

    def step(self, reward, state, action, mask):
        return self._run_one_step(state, reward, action, mask)

    def _run_one_step(self, trajectory, reward, action, mask):
        """Executes a single step in the environment.
        Args:
          action: int, the action to perform in the environment.
        Returns:
          The observation, reward, and is_terminal values returned from the
            environment.
        """
        # if self.num_episodes % 10 == 0:
        #     q_values = self._agent.get_q_values()
        #     mask = self._agent.mask
        #     mask_values = self._agent.get_q_values()[0]
        #     mask_values[mask==0] = -inf
        #     self.video_generator.save_env_image(self._environment.render("rgb_array"), 
        #         skills[action] + "[{}]".format(action) + "\n" + str(q_values) + "\n" + str(mask_values)
        # )

        self.decision_count += 1

        if self._clip_rewards:
            reward = np.clip(reward, -1, 1)
        return self._agent.step(reward, trajectory, action, mask)

    def select_action(self, mask):
        return self._agent.select_action(mask)

    def _run_one_phase(self, min_steps, statistics, run_mode_str):
        """Runs the agent/environment loop until a desired number of steps.
        We follow the Machado et al., 2017 convention of running full episodes,
        and terminating once we've run a minimum number of steps.
        Args:
          min_steps: int, minimum number of steps to generate in this phase.
          statistics: `IterationStatistics` object which records the experimental
            results.
          run_mode_str: str, describes the run mode for this agent.
        Returns:
          Tuple containing the number of steps taken in this phase (int), the sum of
            returns (float), and the number of episodes performed (int).
        """
        # self.sum_returns = 0.
        step_count = 0
        sum_returns = 0.
        self.sum_returns = 0
        num_episodes = 0

        while step_count < min_steps:
            episode_length, episode_return = self._run_one_episode()
            statistics.append({
                '{}_episode_lengths'.format(run_mode_str): episode_length,
                '{}_episode_returns'.format(run_mode_str): episode_return
            })
            self.step_count += episode_length
            step_count += episode_length
            self.sum_returns += episode_return
            sum_returns += episode_return
            # self.num_episodes += 1
            num_episodes += 1
            # We use sys.stdout.write instead of logging so as to flush frequently
            # without generating a line break.
            sys.stdout.write('Episode number: {} '.format(num_episodes) +
                             'Steps executed: {} '.format(step_count) +
                             'Episode length: {} '.format(episode_length) +
                             'Return: {}\r'.format(episode_return))
            # print('Episode number: {} '.format(self.num_episodes) +
            #                  'Steps executed: {} '.format(self.step_count) +
            #                  'Episode length: {} '.format(episode_length) +
            #                  'Return: {}\r'.format(episode_return))
            # sys.stdout.flush()
            # if self.num_episodes % 50 == 0:
            #     self._checkpoint_experiment(self.checkpoint_number)
            #     self._log_experiment(self.checkpoint_number, statistics)
            #     self.checkpoint_number += 1

        # return self.step_count, self.sum_returns, self.num_episodes
        return step_count, sum_returns, num_episodes

    def _initialize_episode(self, initial_observation, mask):
        return self._agent.begin_episode([initial_observation], mask)

    def initialize_episode(self, initial_observation, mask):
        return self._agent.begin_episode(initial_observation, mask)

    def _end_episode(self, reward, action, mask, terminal=True):
        if isinstance(self._agent, jax_dqn_agent.JaxDQNAgent):
            self._agent.end_episode(reward, action, terminal)
        else:
            self._agent.end_episode(reward, action, mask)

        self.episode += 1

    def end_episode(self, reward, action, mask, terminal=True):
        return self._end_episode(reward, action, mask, terminal)

    def reward_function(self, observation):
        return self._agent._get_intrinsic_reward(np.array(observation).reshape((84, 84)))

    def plot(self, episode, steps):
        self._agent.eval_mode = True

        max_range = self.info_buffer.get_last_index()

        rewards = {}
        player_x = {}
        player_y = {}

        for index in range(max_range):
            observation = self.info_buffer.get_index('observation', index)
            room_number = self.info_buffer.get_index('room_number', index)

            if not room_number in rewards:
                rewards[room_number] = []
                player_x[room_number] = []
                player_y[room_number] = []

            rewards[room_number].append(self.reward_function(observation))
            player_x[room_number].append(self.info_buffer.get_index('player_x', index))
            player_y[room_number].append(self.info_buffer.get_index('player_y', index))

        for key in rewards:
            plt.scatter(player_x[key], player_y[key], c=rewards[key], cmap='viridis')
            plt.colorbar()
            figname = self._get_plot_name(self._base_dir, 'reward', str(key), str(episode),
                                          str(steps))
            plt.savefig(figname)
            plt.clf()

        self._agent.eval_mode = False

    def _get_plot_name(self, base_dir, type, room, episode, steps):
        plot_dir = os.path.join(base_dir, 'plots', episode)
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)
        return os.path.join(plot_dir, '{}_room_{}_steps_{}.png'.format(type, room, steps))
