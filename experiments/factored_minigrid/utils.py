import numpy as np
from PIL import Image
import gymnasium as gym
from gymnasium.core import Env, Wrapper, ObservationWrapper
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper, ReseedWrapper, StateBonus
from enum import IntEnum

class actions(IntEnum):
    LEFT        = 0
    RIGHT       = 1
    FORWARD     = 2
    PICKUP      = 3
    DROP        = 4
    TOGGLE      = 5
    DONE        = 6

class FactoredMinigridInfoWrapper(Wrapper):
    
    def __init__(self, env, seed=None):
        super().__init__(env)
        self._timestep = 0
        
        self.env_seed = seed 
    
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
        info['needs_reset'] = truncated # for pfrl
        info['timestep'] = self._timestep # total steps in env
        info['has_key'] = self._has_key()
        info['door_open'] = determine_is_door_open(self)
        info['seed'] = self.env_seed
        
        return info
    
    def _has_key(self):
        if self.env.unwrapped.carrying is None:
            return False
        if self.env.unwrapped.carrying.type == 'key':
            return True
        
        return False

class RandomStartWrapper(Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)
    
    def reset(self):
        return self.env.reset(random_start=True)

class ResizeObsWrapper(ObservationWrapper):
    """Resize obs to (84, 84)"""
    def observation(self, observation):
        num_channels = observation.shape[2]
        new_obs = np.zeros((84, 84, num_channels))
        for channel_idx in range(num_channels):
            img = Image.fromarray(observation[:,:,channel_idx])
            new_obs[:,:,channel_idx] = np.asarray(img.resize((84, 84), Image.BILINEAR))
        
        return new_obs

class TransposeObsWrapper(ObservationWrapper):
    def observation(self, observation):
        assert len(observation.shape) == 3, observation.shape
        assert observation.shape[-1] < observation.shape[0] and observation.shape[-1] < observation.shape[1]
        observation = observation.transpose((2,0,1))
        return observation

class ScaleObsWrapper(ObservationWrapper):
    def observation(self, observation):
        return observation/255.0

def determine_is_door_open(env):
    """Convinence hacky function to determine the goal location."""
    from minigrid.core.world_object import Door
    for i in range(env.grid.width):
        for j in range(env.grid.height):
            tile = env.grid.get(i, j)
            if isinstance(tile, Door):
                return tile.is_open
            
def environment_builder(level_name='FactoredMiniGrid-DoorKey-16x16-v0',
                        scale_obs=True,
                        seed=42,
                        random_reset=False,
                        max_steps=None):
    if max_steps is not None and max_steps > 0:
        env = gym.make(level_name, 
                       max_steps=max_steps,
                       render_mode="rgb_array")
    else:
        env = gym.make(level_name,
                       render_mode="rgb_array")
    env = ReseedWrapper(env, seeds=[seed])
    env = ImgObsWrapper(env)
    env = ResizeObsWrapper(env)
    env = TransposeObsWrapper(env)
    if scale_obs:
        env = ScaleObsWrapper(env)
    if random_reset:
        env = RandomStartWrapper(env) # This is not working right now because of reseeding need to fix
    env = FactoredMinigridInfoWrapper(env, seed)
    
    return env

