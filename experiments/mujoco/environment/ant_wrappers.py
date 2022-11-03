import numpy as np

from experiments.mujoco.environment.vec_env import VecEnvObservationWrapper


class DoubleToFloatWrapper(VecEnvObservationWrapper):
    def process(self, obs):
        return obs.astype(np.float32)
