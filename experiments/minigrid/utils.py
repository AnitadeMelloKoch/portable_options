import math
import pickle
from typing import Tuple
import numpy as np
from PIL import Image
import gymnasium as gym
from gymnasium.core import Env, Wrapper, ObservationWrapper
from minigrid.wrappers import ImgObsWrapper, ReseedWrapper
from enum import IntEnum
import collections
from enum import IntEnum
from gymnasium import spaces


class actions(IntEnum):
    LEFT        = 0
    RIGHT       = 1
    FORWARD     = 2
    PICKUP      = 3
    DROP        = 4
    TOGGLE      = 5
    DONE        = 6

class MinigridInfoWrapper(Wrapper):
    """Include extra information in the info dict for debugging/visualizations."""

    def __init__(self, env, seed=None):
        super().__init__(env)
        self._timestep = 0

        # Store the test-time start state when the environment is constructed
        # self.official_start_obs, self.official_start_info = self.reset()
        self.env_seed = seed
        self.official_start_obs, self.official_start_info = self.reset()

    def reset(self):
        obs, info = self.env.reset(seed=self.env_seed)
        info = self._modify_info_dict(info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._timestep += 1
        info = self._modify_info_dict(info, terminated, truncated)
        done = terminated or truncated
        return obs, reward, done, info

    def _modify_info_dict(self, info, terminated=False, truncated=False):
        info['player_pos'] = tuple(self.env.agent_pos)
        info['player_x'] = self.env.agent_pos[0]
        info['player_y'] = self.env.agent_pos[1]
        info['truncated'] = truncated
        info['terminated'] = terminated
        info['needs_reset'] = truncated  # pfrl needs this flag
        info['timestep'] = self._timestep # total number of timesteps in env
        info['door_open'] = determine_is_door_open(self)
        info['seed'] = self.env_seed
        return info

class FactoredObsWrapperDoorKey(Wrapper):
    def __init__(self, env: Env, type: int=1):
        super().__init__(env)
        self.colours = {
            "blue": 0,
            "red": 1,
            "green": 2,
            "grey": 3,
            "yellow": 4,
            "purple": 5
        }
        self.type = type
    
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        if self.type == 1:
            obs = self._get_factored_obs1()
        if self.type == 2:
            obs = self._get_factored_obs2()
        return obs, info
    
    def _get_factored_obs1(self):
        split_idx = self.env.unwrapped.splitIdx
        
        objects = {
            "door": [-1,-1,-1,-1,-1],
            "key1": [-1, -1, -1],
            "key2": [-1, -1, -1],
            "key3": [-1, -1, -1],
            "key4": [-1, -1, -1],
            "key5": [-1, -1, -1],
            "agent": [-1,-1,-1],
            "goal": [-1, -1],
            "split": [-1]
        }
        
        key_num = 1
        
        for x in range(self.env.unwrapped.width):
            for y in range(self.env.unwrapped.height):
                cell = self.env.unwrapped.grid.get(x,y)
                if cell:
                    if cell.type == "door":
                        objects["door"] = [x, y, int(cell.is_locked), int(cell.is_open), self.colours[cell.color]]
                    if cell.type == "key":
                        objects["key{}".format(key_num)] = [x, y, self.colours[cell.color]]
                        key_num += 1
                    if cell.type == "goal":
                        objects["goal"] = [x, y]
                
        agent_pos = self.env.unwrapped.agent_pos
        agent_dir = self.env.unwrapped.agent_dir
        
        objects["agent"] = [agent_pos[0], agent_pos[1], agent_dir]
        objects["split"] = [split_idx]
        
        factored_obs = []
        
        for key in objects:
            factored_obs += objects[key]
        
        factored_obs = np.array(factored_obs)
        
        return factored_obs
    
    def _get_factored_obs2(self):
        split_idx = self.env.unwrapped.splitIdx
        
        objects = {
            "door": [-1,-1,-1,-1,-1],
            "blue": [-1,-1],
            "red": [-1,-1],
            "green": [-1,-1],
            "grey": [-1,-1],
            "yellow": [-1,-1],
            "purple": [-1,-1],
            "agent": [-1,-1,-1],
            "goal": [-1, -1],
            "split": [-1]
        }
        
        for x in range(self.env.unwrapped.width):
            for y in range(self.env.unwrapped.height):
                cell = self.env.unwrapped.grid.get(x,y)
                if cell:
                    if cell.type == "door":
                        objects["door"] = [x, y, int(cell.is_locked), int(cell.is_open), self.colours[cell.color]]
                    if cell.type == "key":
                        objects[cell.color] = [x, y]
                    if cell.type == "goal":
                        objects["goal"] = [x, y]
                
        agent_pos = self.env.unwrapped.agent_pos
        agent_dir = self.env.unwrapped.agent_dir
        
        objects["agent"] = [agent_pos[0], agent_pos[1], agent_dir]
        objects["split"] = [split_idx]
        
        factored_obs = []
        
        for key in objects:
            factored_obs += objects[key]
        
        factored_obs = np.array(factored_obs)
        
        return factored_obs
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if self.type == 1:
            factored_obs = self._get_factored_obs1()
        if self.type == 2:
            factored_obs = self._get_factored_obs2()
        
        return factored_obs, reward, terminated, truncated, info

class ResizeObsWrapper(ObservationWrapper):
    """Resize the observation image to be (84, 84) and compatible with Atari."""
    def observation(self, observation):
        img = Image.fromarray(observation)
        return np.asarray(img.resize((84, 84), Image.BILINEAR))


class TransposeObsWrapper(ObservationWrapper):
    def observation(self, observation):
        assert len(observation.shape) == 3, observation.shape
        assert observation.shape[-1] == 3, observation.shape
        return observation.transpose((2, 0, 1))


class SparseRewardWrapper(Wrapper):
    """Return a reward of 1 when you reach the goal and 0 otherwise."""
    def step(self, action):
        # minigrid discounts the reward with a step count - undo that here
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, float(reward > 0), terminated, truncated, info


class GrayscaleWrapper(ObservationWrapper):
    def observation(self, observation):
        observation = observation.mean(axis=0)[np.newaxis, :, :]
        return observation.astype(np.uint8)

class ScaleObsWrapper(ObservationWrapper):
    def __init__(self, 
                 env: Env,
                 image_size: tuple):
        super().__init__(env)
        
        self.image_size = image_size
    
    def observation(self, observation):
        img = Image.fromarray(observation)
        return np.asarray(img.resize(self.image_size, Image.BICUBIC))

class NormalizeObsWrapper(ObservationWrapper):
    def observation(self, observation):
        observation = observation/255.0
        return observation

class PadObsWrapper(ObservationWrapper):
    def __init__(self, 
                 env: Env,
                 image_size: tuple):
        super().__init__(env)
        
        self.image_size = image_size
    
    def observation(self, observation):
        x, y, c = observation.shape
        pad_x = (self.image_size[0] - x) // 2
        pad_y = (self.image_size[1] - y) // 2
        
        if pad_x%2 == 0:
            final_pad_x = (pad_x, pad_x)
        else:
            final_pad_x = (pad_x + 1, pad_x)
        
        if pad_y%2 == 0:
            final_pad_y = (pad_y, pad_y)
        else:
            final_pad_y = (pad_y + 1, pad_y)
        
        padded_obs = np.stack([
            np.pad(observation[:,:,idx], (final_pad_x, final_pad_y), mode="constant", constant_values=0) for idx in range(3)
        ], axis=-1)
                
        return padded_obs

class RandomStartWrapper(Wrapper):
    def __init__(self, env, start_locs=[]):
        
        super().__init__(env)
        self.n_episodes = 0
        self.start_locations = start_locs

        # TODO(ab): This assumes that the 2nd-to-last action is unused in the env
        # Not using the last action because that terminates the episode!
        self.no_op_action = env.action_space.n - 2

    def reset(self):
        super().reset()
        rand_pos = self.start_locations[self.n_episodes % len(self.start_locations)]
        self.n_episodes += 1
        return self.reset_to(rand_pos)

    def reset_to(self, rand_pos):
        new_pos = self.env.place_agent(
        top=rand_pos,
        size=(1, 1)
        )

        # Apply the no-op to get the observation image
        obs, _, _, info = self.env.step(self.no_op_action)

        info['player_x'] = new_pos[0]
        info['player_y'] = new_pos[1]
        info['player_pos'] = new_pos
        
        return obs, info


def determine_goal_pos(env):
    """Convinence hacky function to determine the goal location."""
    from minigrid.core.world_object import Goal
    for i in range(env.grid.width):
        for j in range(env.grid.height):
            tile = env.grid.get(i, j)
            if isinstance(tile, Goal):
                return i, j


def determine_is_door_open(env):
    """Convinence hacky function to determine the goal location."""
    from minigrid.core.world_object import Door
    for i in range(env.grid.width):
        for j in range(env.grid.height):
            tile = env.grid.get(i, j)
            if isinstance(tile, Door):
                return tile.is_open

class RGBImgObsWrapper(ObservationWrapper):
    """
    Wrapper to use fully observable RGB image as observation,
    This can be used to have the agent to solve the gridworld in pixel space.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width * tile_size, self.env.height * tile_size, 3),
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        rgb_img = self.get_frame(highlight=False, tile_size=self.tile_size)

        return {**obs, "image": rgb_img}

def environment_builder(
    level_name='MiniGrid-Empty-8x8-v0',
    reward_fn='sparse',
    grayscale=True,
    scale_obs=False,
    pad_obs=False,
    add_count_based_bonus=True,
    exploration_reward_scale=0,
    seed=42,
    random_reset=False,
    max_steps=None,
    random_starts=[],
    final_image_size=(152,152),
    normalize_obs=True
    ):
    if max_steps is not None and max_steps > 0:
        env = gym.make(level_name, max_steps=max_steps,
                       render_mode="rgb_array")  #, goal_pos=(11, 11))
    else:
        env = gym.make(level_name,
                       render_mode="rgb_array")
    # env = ReseedWrapper(env, seeds=[seed])  # To fix the start-goal config
    env = RGBImgObsWrapper(env) # Get pixel observations
    env = ImgObsWrapper(env) # Get rid of the 'mission' field
    if normalize_obs is True:
        env = NormalizeObsWrapper(env)
    if reward_fn == 'sparse':
        env = SparseRewardWrapper(env)
    # if scale_obs is True:
    #     env = ScaleObsWrapper(env, final_image_size)
    if pad_obs is True:
        env = PadObsWrapper(env, final_image_size)
    env = TransposeObsWrapper(env)
    if grayscale is True:
        env = GrayscaleWrapper(env)
    env = MinigridInfoWrapper(env, seed)
    if random_reset is True:
        assert exploration_reward_scale == 0, exploration_reward_scale
        assert len(random_starts) > 0
        env = RandomStartWrapper(env, random_starts)
    return env

def process_data(array):
    d = collections.OrderedDict()
    d["mean"] = np.mean(array)
    d["std"] = np.std(array)
    d["min"] = np.amin(array)
    d["max"] = np.amax(array)
    return d

def factored_environment_builder(level_name='AdvancedDoorKey-8x8-v0',
                                 seed=42,
                                 max_steps=None,
                                 factored_type=1):
    if max_steps is not None and max_steps > 0:
        env = gym.make(level_name, max_steps=max_steps,
                       render_mode="rgb_array")
    else:
        env = gym.make(level_name,
                       render_mode="rgb_array")
    env = ReseedWrapper(env, seeds=[seed])
    env = FactoredObsWrapperDoorKey(env, type=factored_type)
    env = MinigridInfoWrapper(env, seed)
    
    return env

