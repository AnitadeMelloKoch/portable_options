import numpy as np

from .vec_env import VecEnvRewardWrapper


class VecClipRewards(VecEnvRewardWrapper):
    """clip the rewards to prevent explosion"""
    def __init__(self, venv, clip_value=1.0):
        super().__init__(venv=venv)
        self.clip_value = clip_value

    def process(self, rewards):
        return np.clip(rewards, -self.clip_value, self.clip_value)
        