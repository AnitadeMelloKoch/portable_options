from experiments.procgen.core.procgen_experiment import ProcgenExperiment
from procgen import ProcgenEnv
import torch

from experiments.procgen.core.vec_env import VecExtractDictObs, VecNormalize, VecChannelOrder, VecMonitor
from portable.option.ensemble.custom_attention import AutoEncoder

import matplotlib.pyplot as plt 
import numpy as np

def embedding_phi(x, use_gpu):
    x = x/255.
    x = torch.from_numpy(x).float()
    if use_gpu:
        x = x.to("cuda")
    return x

def make_vector_env(level_index):
    """
    make a procgen environment such that the training env only focus on 1 particular level
    the testing env will sample randomly from a number of levels

    NOTE: Warning: the eval environment does not guarantee sequential transition between `num_levels` levels.
    The following level is sampled randomly, so evaluation is not deterministic, and performed on god knows
    which levels.
    """
    
    venv = ProcgenEnv(num_envs=1,
                        env_name="coinrun",
                        num_levels=1,
                        start_level=level_index,
                        distribution_mode="easy",
                        num_threads=4,
                        center_agent=True,
                        rand_seed=0)
    
    venv = VecChannelOrder(venv, channel_order="chw")
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv, ob=False)
    
    return venv

autoencoder = AutoEncoder(num_input_channels=3,
                          feature_size=500,
                          image_height=64,
                          image_width=64).to("cuda")
autoencoder.load_state_dict(torch.load("resources/encoders/procgen/coinrun.ckpt"))

env = make_vector_env(27)

obs = env.reset()

obs = embedding_phi(obs, True)

print(obs)

fig, axes = plt.subplots(ncols=2)
sample = obs[0]
with torch.no_grad():
    axes[0].set_axis_off()
    axes[1].set_axis_off()
    with torch.no_grad():
        pred = autoencoder(sample.unsqueeze(0)).cpu().numpy()
    axes[0].imshow(np.transpose(sample.cpu().numpy(), axes=(1,2,0)))
    axes[1].imshow(np.transpose(pred[0], axes=(1,2,0)))

    plt.show()





