from experiments.core.divdis_meta_experiment import DivDisMetaExperiment
import argparse 
from portable.utils.utils import load_gin_configs
import torch 
from experiments.minigrid.utils import environment_builder
from portable.agent.model.ppo import create_cnn_policy, create_cnn_vf
import numpy as np
import matplotlib.pyplot as plt

env = environment_builder('LockedRoom-v0',
                          seed=0,
                          grayscale=False,
                          normalize_obs=False,
                          scale_obs=True,
                          final_image_size=(94,94))

obs, _ = env.reset()

plt.imshow(obs.transpose(1,2,0))
plt.show()



