import numpy as np
from pfrl.agents import SoftActorCritic


class SAC(SoftActorCritic):
    """
    custom SAC so that it can be used for procgen envs
    """
    def batch_act(self, batch_obs):
        return np.array(super().batch_act(batch_obs))
