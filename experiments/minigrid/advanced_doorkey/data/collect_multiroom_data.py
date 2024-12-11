from experiments.minigrid.utils import environment_builder
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
from experiments.minigrid.utils import actions 

from gymnasium.core import Env, Wrapper
from experiments.minigrid.advanced_doorkey.data.collect_lockedroom_data import DataCollectorWrapper


states = []

for x in tqdm(range(40000)):

    env = environment_builder('MiniGrid-MultiRoom-N6-v0',
                            seed=np.random.randint(0, 10000),
                            grayscale=False)
    obs, _ = env.reset()

    states.append(obs)
    
    # fig, axes = plt.subplots()
    # axes.imshow(np.transpose(obs, axes=(1,2,0)))
    # plt.show()

print(len(states))

np.save("resources/minigrid_images/multiroom_random_states.npy", states)








