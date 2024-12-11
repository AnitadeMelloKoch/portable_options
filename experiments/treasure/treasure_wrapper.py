from typing import Any, SupportsFloat
from gymnasium.core import Env, Wrapper
import torch

class TreasureInfoWrapper(Wrapper):
    def __init__(self, env: Env, timelimit = 10000):
        super().__init__(env)
        self._timestep = 0
        self.timelimit = timelimit
    
    def reset(self):
        obs = self.env.reset()
        obs = torch.tensor(obs)
        self._timestep = 0
        
        return obs, {}
    
    def step(self, action: Any):
        obs, reward, done, info = self.env.step(action)
        self._timestep += 1
        if self._timestep >= self.timelimit:
            done = True
        return torch.tensor(obs), reward, done, info