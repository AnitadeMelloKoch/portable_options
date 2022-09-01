from pathlib import Path

import numpy as np
from gym import Wrapper

from portable.ale_utils import set_player_ram


class MonteForwarding(Wrapper):
    """
    forwards the agent to another state when the agent starts
    this just overrides the reset method and make it start in another position

    Note: because of the reset() method, this must go immediately after the FrameStack wrapper
    """
    def __init__(self, env, forwarding_target: Path):
        """
        forward the agent to start in state `forwarding_target`
        Args:
            forwarding_target: a previously saved .npy file that contains the encoded start state ram
        """
        super().__init__(env)
        self.env = env
        self.target_ram = np.load(forwarding_target)
        self.reset()  # reset from init so that the agent starts in the correct position
        # the monte ladder goal wrapper depends on agent being in the right room when env is created
    
    def reset(self):
        self.env.reset()
        obs = set_player_ram(self.env, self.target_ram)
        # because we are calling reset here, self.env.unwrapped.original_frame_stack is recording the first room
        # extra steps so that the original framestacks of the forwarding room cover up the remnants of the first room
        for _ in range(4):
            self.env.step(0)
        obs = self.env._get_ob()  # method from FrameStack
        return obs
