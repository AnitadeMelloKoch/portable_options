from experiments.minigrid.advanced_doorkey.core.policy_train_wrapper import AdvancedDoorKeyPolicyTrainWrapper
import numpy as np 
import matplotlib.pyplot as plt 
from experiments.minigrid.utils import environment_builder
from tqdm import tqdm 
import random

key = [False, True, True, True]
door_unlocked = [False, False, True, True]
door_open = [False, False, False, True]

colours = ["red", "green", "blue", "purple", "yellow", "grey"]

# fig, ax = plt.subplots()

states = []

for x in range(4):
    for y in tqdm(range(2000)):
        for colour in colours:
            remaining_col = list(filter(lambda c: c != colour, colours))
            random.shuffle(remaining_col)
            seed = np.random.randint(low=0, high=1000)
            env = environment_builder('AdvancedDoorKey-16x16-v0',
                                      seed=seed,
                                      grayscale=False)
            env = AdvancedDoorKeyPolicyTrainWrapper(env,
                                                    key_collected=key[x],
                                                    door_unlocked=door_unlocked[x],
                                                    door_open=door_open[x],
                                                    door_colour=colour,
                                                    key_colours=remaining_col)
            
            obs, _ = env.reset(agent_reposition_attempts=np.random.randint(low=1, high=20))
            
            states.append(obs)

print(len(states))

np.save("resources/minigrid_images/adv_doorkey_random_states.npy", states)





