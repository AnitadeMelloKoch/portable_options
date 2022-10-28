import numpy as np
from gym import spaces

from .vec_env import VecEnvObservationWrapper


class VecChannelOrder(VecEnvObservationWrapper):
    def __init__(self, venv, channel_order="chw"):
        """change the channel order of the observation"""
        self.width = 64
        self.height = 64
        self.channel_order = channel_order
        shape = {
            'hwc': (self.height, self.width, 3),
            'chw': (3, self.height, self.width),
        }
        observation_box = spaces.Box(
            low=0, high=255, shape=shape[channel_order], dtype=np.uint8
        )
        observation_space = venv.observation_space
        observation_space.spaces ['rgb'] = observation_box
        super().__init__(venv=venv, observation_space=observation_space)

    def process(self, obs):
        """
        obs is a dict with key 'rgb'
        the value is np array of shape (num_envs, height, width, 3)
        """
        order = (0,1,2,3) if self.channel_order == "hwc" else (0,3,1,2)
        obs['rgb'] = obs['rgb'].transpose(order)
        return obs
