from experiments.minigrid.advanced_doorkey.core.policy_train_wrapper import AdvancedDoorKeyPolicyTrainWrapper
import numpy as np 
import matplotlib.pyplot as plt 
from experiments.minigrid.utils import environment_builder, factored_environment_builder
from tqdm import tqdm 
import random

colours = ["red", "green", "blue", "purple", "yellow", "grey"]
# colours = ["red", "green", "blue", "purple", "yellow"]

# fig, ax = plt.subplots()


seed = 9

states = []
for colour in colours:
    for y in tqdm(range(50)):
        env = environment_builder('AdvancedDoorKey-8x8-v0',
                                            seed=seed,
                                            grayscale=False)
        env = AdvancedDoorKeyPolicyTrainWrapper(env,
                                                door_colour=colour,
                                                image_input=False)
        
        obs, _ = env.reset(random_start=True,
                        keep_colour=colour,
                        agent_position=(6,6))
        
        states.append(obs.numpy())

states = np.array(states)

print(states.shape)
print(len(states))

np.save("resources/minigrid_images/adv_doorkey_8x8_v2_togoal_{}_termination_positive.npy".format(seed)
        , states)





