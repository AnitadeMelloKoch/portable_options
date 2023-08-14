import math
import pickle
import numpy as np
from PIL import Image
import gymnasium as gym
from gymnasium.core import Env, Wrapper, ObservationWrapper
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper, ReseedWrapper, StateBonus
from enum import IntEnum
import collections

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
        obs, info = self.env.reset()
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
        info['has_key'] = self.env.unwrapped.carrying is not None
        if info['has_key']:
            assert self.unwrapped.carrying.type == 'key', self.env.unwrapped.carrying
        info['door_open'] = determine_is_door_open(self)
        info['seed'] = self.env_seed
        return info


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
    
class ScaledStateBonus(StateBonus):
    """Slight mod of StateBonus: scale the count-based bonus before adding."""

    def __init__(self, env, reward_scale):
        super().__init__(env)
        self.reward_scale = reward_scale

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Tuple based on which we index the counts
        # We use the position after an update
        env = self.unwrapped
        tup = tuple(env.agent_pos)

        # Get the count for this key
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this key
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / math.sqrt(new_count)
        reward += (self.reward_scale * bonus)

        # Add to the info dict
        info['count'] = new_count
        info['bonus'] = bonus

        return obs, reward, terminated, truncated, info
    

class ScaleObsWrapper(ObservationWrapper):
    def observation(self, observation):
        img = Image.fromarray(observation)
        return np.asarray(img.resize((128, 128), Image.BILINEAR))

class PadObsWrapper(ObservationWrapper):
    def __init__(self, 
                 env: Env,
                 pad: tuple):
        super().__init__(env)
        
        self.pad = pad
    
    def observation(self, observation):
        padded_obs = np.stack([
            np.pad(observation[:,:,idx], self.pad, mode="constant", constant_values=0) for idx in range(3)
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
    random_starts=[]
    ):
    if max_steps is not None and max_steps > 0:
        env = gym.make(level_name, max_steps=max_steps,
                       render_mode="rgb_array")  #, goal_pos=(11, 11))
    else:
        env = gym.make(level_name,
                       render_mode="rgb_array")
    env = ReseedWrapper(env, seeds=[seed])  # To fix the start-goal config
    env = RGBImgObsWrapper(env) # Get pixel observations
    env = ImgObsWrapper(env) # Get rid of the 'mission' field
    if reward_fn == 'sparse':
        env = SparseRewardWrapper(env)
    if scale_obs:
        print("scale obs")
        env = ScaleObsWrapper(env)
    if pad_obs:
        env = PadObsWrapper(env)
    # env = ResizeObsWrapper(env)
    env = TransposeObsWrapper(env)
    if grayscale:
        env = GrayscaleWrapper(env)
    if add_count_based_bonus:
        env = ScaledStateBonus(env, exploration_reward_scale)
    env = MinigridInfoWrapper(env, seed)
    if random_reset:
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



