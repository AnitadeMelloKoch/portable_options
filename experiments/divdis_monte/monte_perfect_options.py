import argparse
from collections import deque
import datetime
import logging
import math
import sys
import numpy as np
import random
import gin
import os
import matplotlib.pyplot as plt 
import torch
import json
from math import floor
import pickle
import gym
from torch.utils.tensorboard import SummaryWriter 

from experiments.experiment_logger import VideoGenerator
# from portable.agent.model.ppo import ActionPPO, create_atari_model, create_linear_atari_model
from portable.agent.model.maskable_ppo import MaskablePPOAgent, create_mask_atari_model, create_mask_linear_atari_model, TabularAgent
from portable.utils.utils import load_gin_configs, set_seed

sys.path.append("../pix2sym")
from gym_montezuma.envs.montezuma_env import make_monte_env_as_atari_deepmind, Option, MontezumaEnv

'''
Takes in a dict representing the privileged state parsed from the RAM of 
Montezuma's Revenge (pix2sym/ataritools/envs/montezumaenv.py) and returns a vector that can be used as input for
a neural network
'''
class PrivilegedStateWrapper(gym.Wrapper):
    categorical = ['player_status', 'player_look', 'door_left', 'door_right', 'object_type', 'object_configuration', 'object_dir', 'skull_dir', 'object_vertical_dir']

    categories = [
        ['standing', 'running', 'on-ladder', 'climbing-ladder', 'on-rope', 'climbing-rope', 'mid-air', 'dead'], # player_status
        ['left', 'right'], # player_look
        ['locked', 'unlocked'], # door_left
        ['locked', 'unlocked'], # door_right
        ['none', 'jewel', 'sword', 'mallet', 'key', 'jump_skull', 'torch', 'snake', 'spider'], # object_type
        ['one_single', 'two_near', 'two_mid', 'three_near', 'two_far', 'one_double', 'three_mid', 'one_triple'], # object_configuration
        ['left', 'right'], # object_dir
        ['left', 'right'], # skull_dir
        ['up', 'down']  # object_vertical_dir
    ]

    bools = ['screen_changing', 'has_ladder', 'has_rope', 'has_lasers', 'has_platforms', 'has_bridge', \
    'respawning', 'has_spider', 'has_snake', 'has_jump_skull', 'has_enemy', 'has_jewel']

    # numerical binary also gets included in numerical
    numerical = ['timestep', 'frame', 'level', 'screen', 'level', 'score', 'time_to_appear', \
    'time_to_disappear', 'player_x', 'player_y', 'respawned', 'player_jumping', \
    'player_falling', 'lives', 'just_died', 'time_to_spawn', 'has_skull', 'has_object', \
    'object_x', 'object_y', 'object_y_offset', 'skull_x']

    possible_items = ['torch', 'sword', 'sword', 'key', 'key', 'key', 'key', 'hammer']

    def __init__(self, env):
        super().__init__(env)
        self.vec_size = len(self.numerical) + \
                        2 * len(self.bools) + \
                        sum([len(cats) for cats in self.categories]) + \
                        len(self.possible_items)
        self._env = env

    def vectorize_state(self, state_dict: dict):
        vec = np.zeros(self.vec_size)

        index = 0

        # numerical features
        for num in self.numerical:
            vec[index] = state_dict[num]
            index += 1

        # bools
        for b in self.bools:
            if state_dict[b]:
                vec[index] = 0
                vec[index + 1] = 1
            else:
                vec[index] = 1
                vec[index] = 0 
            index += 2

        # categorical features
        for i, cat in enumerate(self.categorical):
            hot_i = self.categories[i].index(state_dict[cat])
            vec[index + hot_i] = 1
            index += len(self.categories[i])

        # inventory
        inv = state_dict["inventory"].copy()
        for item in self.possible_items:
            if item in inv:
                vec[index] = 1
                inv.remove(item)
            index += 1
        
        return vec
    
    def reset(self):
        observation = self._env.reset()
        return self.vectorize_state(self._env.getState())
    
    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return self.vectorize_state(self._env.getState()), reward, done, info

'''
Provides intra-life rewards for visiting each xy bin in different rooms
'''
class CuriosityWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.bin_width = 40 # 160 / 40 = grid width of 4
        self.bin_height = 30 # 210 / 30 = grid height of 7
        self._env = env
        self.visit_reward = 0.1
        
    def reset(self):
        observation = self._env.reset()
        self.has_visited = np.zeros((99, 10, 10))
        return observation
    
    def step(self, action):
        observation, reward, done, info = self._env.step(action)
        return observation, self.reward(reward), done, info

    def reward(self, reward):
        x = self._env.getState()['player_x']
        y = self._env.getState()['player_y']
        room = self._env.getState()['level']

        grid_x = floor(x / self.bin_width)
        grid_y = floor(y / self.bin_height)

        # print(room, grid_x, grid_y)

        if self.has_visited[room, grid_y, grid_x] == 0:
            self.has_visited[room, grid_y, grid_x] = 1
            return reward + self.visit_reward

        return reward

@gin.configurable
def make_monte_with_skills_env(game_name=None,
                      sticky_actions=True,
                      episode_life=True,
                      clip_rewards=False,
                      frame_skip=4,
                      frame_stack=4,
                      frame_warp=(84, 84),
                      max_episode_steps=None,
                      single_life=False,
                      single_screen=False,
                      seed=None,
                      noop_wrapper=False,
                      render_option_execution=False,
                      expose_primitive_actions=False):

    d = {
        "single_life": single_life,
        "single_screen": single_screen,
        "seed": seed,
        "noop_wrapper": noop_wrapper,
        "render_option_execution": render_option_execution,
        "expose_primitive_actions": expose_primitive_actions
    }

    env = make_monte_env_as_atari_deepmind(max_episode_steps=max_episode_steps,
                                           episode_life=episode_life,
                                           clip_rewards=clip_rewards,
                                           frame_skip=frame_skip,
                                           frame_stack=frame_stack,
                                           frame_warp=frame_warp,
                                           **d)
    return env

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)

# SEQUENCE_TO_EXECUTE = [1, 9, 19, 18]
SEQUENCE_TO_EXECUTE = [6, 2, 2, 10, 10, 6, 1]

# log sequence of options that lead to timeouts
# then log every step of states/screenshots for timeout sequences

# make it continually get to the key, check the probs
# if it's choosing correct actions but not getting to the goal, skills are broken somewhere

@gin.configurable
class MontePerfectOptionsExperiment:
    def __init__(self,
                 base_dir,
                 experiment_name,
                 seed,
                 agent_phi,
                 use_gpu,
                 num_actions,
                 env,
                 gpu_list=[0],
                 use_privileged_state=False,
                 add_curiosity=False,
                 action_model=None,
                 discount_rate=0.9,
                 make_videos=False):
        self.name = experiment_name
        self.seed = seed 
        self.use_gpu = use_gpu
        
        self.process_obs = agent_phi
        
        self.base_dir = os.path.join(base_dir, experiment_name, str(seed))
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.save_dir = os.path.join(self.base_dir, 'checkpoints')
        self.plot_dir = os.path.join(self.base_dir, 'plots')
        
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        self.decisions = 0            
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        
        if add_curiosity:
            env = CuriosityWrapper(env)
        
        self.use_privileged_state = use_privileged_state
        if use_privileged_state:
            env = PrivilegedStateWrapper(env)

        self.env = env

        set_seed(seed)
        
        log_file = os.path.join(self.log_dir, 
                                "{}.log".format(str(datetime.datetime.now()).replace(":", "_")))
        logging.basicConfig(filename=log_file, 
                            format='%(asctime)s %(levelname)s: %(message)s',
                            level=logging.INFO)
        logging.info("[experiment] Beginning experiment {} seed {}".format(self.name, self.seed))
        
        if make_videos:
            self.video_generator = VideoGenerator(os.path.join(self.base_dir, "videos"))
        else:
            self.video_generator = None
        
        self.num_options = len(Option.__members__)

        model = action_model if not use_privileged_state else create_mask_linear_atari_model(91, num_actions)

        self.meta_agent = MaskablePPOAgent(use_gpu=gpu_list[-1],
                                           model=model,
                                           phi=agent_phi)

        
        self._cumulative_discount_vector = np.array(
            [math.pow(discount_rate, n) for n in range(100)]
        )

        self.gamma = discount_rate
        
        self.experiment_data = []
        self.episode_data = []
        
    def get_option_mask(self, env):
        options_mask = env.available_options() > 0
        options_mask = options_mask[np.newaxis, :]
        return options_mask
    
    def act(self, obs, mask):
        action = self.meta_agent.act(obs, mask)
        return action
    
    def get_option_name(self, option_num):
        return self.env.get_action_meanings()[option_num]
    
    def _video_log(self, line):
        if self.video_generator is not None:
            self.video_generator.add_line(line)
    
    def save(self):
        with open(os.path.join(self.save_dir, "experiment_results.pkl"), 'wb') as f:
            pickle.dump(self.experiment_data, f)
        
        with open(os.path.join(self.save_dir, "episode_results.pkl"), 'wb') as f:
            pickle.dump(self.episode_data, f)
        
        np.save(os.path.join(self.save_dir, "decisions.npy"), self.decisions)
    
    def save_ram_and_screenshot(self, env, option, step, dir_name, trajectory = None):
        parent_dir = f'./{dir_name}/{self.get_option_name(option)}'

        i = 0
        while os.path.exists(os.path.join(parent_dir, str(i))):
            i += 1
        
        if i > 100:
            return
        
        dir_path = os.path.join(parent_dir, str(i))
        os.makedirs(dir_path, exist_ok=True)

        # save RAM state
        with open(os.path.join(dir_path, 'state.json'), "wt") as f:
            # f.write(str(env.getState()))
            state = env.getState()
            del state['env']
            json.dump(state, f, indent=4, cls=NumpyEncoder)

        # save initiation screenshot
        img = np.squeeze(env.render("rgb_array"))
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.imshow(img)
        ax1.axis('off')
        ax2.axis('off')
        fig.savefig(os.path.join(dir_path, "screenshot.png"))
        plt.close(fig)

        if trajectory:
            with open(os.path.join(dir_path, 'trajectory.txt'), 'wt') as f:
                f.write(str(trajectory))

    def observe(self, 
                obs,
                mask, 
                rewards, 
                done):
        
        if len(rewards) > len(self._cumulative_discount_vector):
            self._cumulative_discount_vector = np.array(
                [math.pow(self.gamma, n) for n in range(len(rewards))]
            )
        
        reward = np.sum(self._cumulative_discount_vector[:len(rewards)]*rewards)
        
        self.meta_agent.observe(obs,
                                mask,
                                reward,
                                done,
                                done)

    def save_image(self, env):
        if self.video_generator is not None:
            img = env.render("rgb_array")
            self.video_generator.make_image(img)

    def train_meta_agent(self,
                         seed,
                         max_steps,
                         min_performance=1.0):
        total_steps = 0
        episode_rewards = deque(maxlen=200)
        episode = 0
        undiscounted_rewards = []
        trajectories = []

        while total_steps < max_steps:
            undiscounted_reward = 0         # episode reward
            done = False
            
            if self.video_generator is not None:
                self.video_generator.episode_start()
            
            obs = self.env.reset()
            if not self.use_privileged_state:
                obs = self.process_obs(obs)

            trajectories.append([])

            current_steps = 0

            while not done:
                self.save_image(self.env)
                if type(obs) == np.ndarray:
                    obs = torch.from_numpy(obs).float()
                option_mask = self.get_option_mask(self.env)

                action, q_values = self.act(obs, option_mask)
                trajectories[-1].append(action)
                
                self._video_log("action: {}".format(action))
                # self._video_log("available actons: {}".format(option_mask))
                # self._video_log("q values: {}".format(q_values))
                
                # self.save_ram_and_screenshot(self.env, SEQUENCE_TO_EXECUTE[current_steps], total_steps, 'seq')

                # action = SEQUENCE_TO_EXECUTE[current_steps]
                print(f'available actions: {np.argwhere(self.get_option_mask(self.env))[:,1]}')
                action = int(input('enter integer option to execute (0-26): '))
                print(f"executing {self.get_option_name(action)}...")

                next_obs, reward, done, info = self.env.step(action)

                # if total_steps == len(SEQUENCE_TO_EXECUTE) - 1:
                #     self.video_generator.episode_end("episode_{}".format(episode))

                # if info['timeout']:
                #     self.save_ram_and_screenshot(self.env, action, total_steps, 'timeouts', trajectories[-6:])

                if not self.use_privileged_state:
                    next_obs = self.process_obs(next_obs)

                frames, rewards, actions, steps = info['frames'], info['rewards'], info['actions'], info['n_steps']
                undiscounted_reward += reward
                
                self.decisions += 1
                total_steps += steps
                current_steps += 1
                
                self.experiment_data.append({
                    "meta_step": self.decisions,
                    "option_length": steps,
                    "option_rewards": rewards,
                    "frames": total_steps
                })
                
                self.observe(obs,
                             option_mask,
                             rewards,
                             done)
                obs = next_obs
            
            logging.info("Episode {} total steps: {} decisions: {}  average undiscounted reward: {} epsiode reward: {}".format(episode,
                                                                                     total_steps,
                                                                                     self.decisions,  
                                                                                     np.mean(episode_rewards),
                                                                                     undiscounted_reward))
            
            if (undiscounted_reward > 0 or episode%10==0) and self.video_generator is not None:
                self.video_generator.episode_end("episode_{}".format(episode))
            
            undiscounted_rewards.append(undiscounted_reward)
            episode += 1
            episode_rewards.append(undiscounted_rewards)
            
            self.episode_data.append({
                "episode": episode,
                "episode_rewards": undiscounted_reward,
                "frames": total_steps
            })
            
            self.writer.add_scalar('episode_rewards', undiscounted_reward, total_steps)
                        
            if episode % 50 == 0:
                self.meta_agent.save(os.path.join(self.save_dir, "action_agent"))
                self.save()
            
            # if total_steps > 1e6 and np.mean(episode_rewards) > min_performance:
            #     logging.info("Meta agent reached min performance {} in {} steps".format(np.mean(episode_rewards),
            #                                                                             total_steps))
            #     return
        
    def eval_meta_agent(self,
                        seed,
                        num_runs):
        undiscounted_rewards = []
        
        with self.meta_agent.agent.eval_mode():
            for run in range(num_runs):
                total_steps = 0
                undiscounted_reward = 0
                done = False
                if self.video_generator is not None:
                    self.video_generator.episode_start()
                obs = self.env.reset()

                if not self.use_privileged_state:
                    obs = self.process_obs(obs)


                while not done:
                    self.save_image(self.env)
                    if type(obs) == np.ndarray:
                        obs = torch.from_numpy(obs).float()
                    option_mask = self.get_option_mask(env)
                    action, q_values = self.act(obs, option_mask)
                    
                    self._video_log("[meta] action: {}".format(action))
                    # self._video_log("available actons: {}".format(np.argwhere(option_mask.reshape(-1))))
                    self._video_log("q_values: {}".format(q_values))
                    
                    next_obs, reward, done, info = self.env.step(action)

                    if not self.use_privileged_state:
                        next_obs = self.process_obs(next_obs)

                    frames, rewards, actions, steps = info['frames'], info['rewards'], info['actions'], info['n_steps']
                    undiscounted_reward += reward
                    total_steps += steps
                    
                    self.observe(obs,
                                    option_mask,
                                    rewards,
                                    done)
                    obs = next_obs
                
                logging.info("Eval {} total steps: {} undiscounted reward: {}".format(run,
                                                                                      total_steps,
                                                                                      undiscounted_reward))
            
                if self.video_generator is not None:
                    print("episode end - wrote video")
                    self.video_generator.episode_end("eval_{}".format(run))
                
                undiscounted_rewards.append(undiscounted_reward)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config_file", nargs='+', type=str, required=True)
    parser.add_argument("--gin_bindings", default=[], help='Gin bindings to override the values' + 
            ' set in the config files (e.g. "DQNAgent.epsilon_train=0.1",' +
            ' "create_atari_environment.game_name="Pong"").')
    
    args = parser.parse_args()
    load_gin_configs(args.config_file, args.gin_bindings)
    
    def process_obs(x):
        x = np.array(x) / 255.0
        return torch.from_numpy(x).float()

    frame_stack = 4

    env = make_monte_with_skills_env(seed=args.seed, frame_stack=frame_stack)

    print("available actions:", env.action_space.n)

    experiment = MontePerfectOptionsExperiment(base_dir=args.base_dir,
                                      seed=args.seed,
                                      agent_phi=process_obs,
                                      action_model=create_mask_atari_model(frame_stack, env.action_space.n),
                                      make_videos=False,
                                      num_actions=env.action_space.n,
                                      env=env)


    # meta_agent_load_dir = "runs/monte_meta_perfect_options/8/checkpoints/action_agent"
    # print('about to load agent', experiment.meta_agent)
    # experiment.meta_agent.load(meta_agent_load_dir)
    # print('just loaded agent')

    # experiment.eval_meta_agent(num_runs=100,
    #                             seed=args.seed)

    experiment.train_meta_agent(max_steps=4e7,
                                seed=args.seed)
