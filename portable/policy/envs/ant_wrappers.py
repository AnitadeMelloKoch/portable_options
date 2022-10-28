import numpy as np

from portable.policy.vec_env.vec_env import VecEnvObservationWrapper


class DoubleToFloatWrapper(VecEnvObservationWrapper):
    def process(self, obs):
        return obs.astype(np.float32)
