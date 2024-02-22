from experiments.minigrid.advanced_doorkey.core.policy_train_wrapper import AdvancedDoorKeyPolicyTrainWrapper
import numpy as np 
import matplotlib.pyplot as plt 
from experiments.minigrid.utils import environment_builder, factored_environment_builder
from tqdm import tqdm 
import random

colours = ["red", "green", "blue", "purple", "yellow", "grey"]
# colours = ["red", "green", "blue", "purple", "yellow"]

# fig, ax = plt.subplots()


seed = 11

for colour in colours:
    states = []
    for y in tqdm(range(2000)):
        repos_attempts = np.random.randint(low=0, high=1000)
        env = factored_environment_builder('AdvancedDoorKey-8x8-v0',
                                            seed=seed)
        env = AdvancedDoorKeyPolicyTrainWrapper(env,
                                                door_colour=colour,
                                                image_input=False)
        
        obs, _ = env.reset(random_start=True,
                        keep_colour=colour,
                        agent_reposition_attempts=repos_attempts)
        
        states.append(obs.numpy())

    states = np.array(states)
    
    print(states.shape)
    print(len(states))

    np.save("resources/factored_minigrid_images/adv_doorkey_8x8_v2_get{}key_door{}_{}_1_termination_negative.npy".format(colour, colour, seed)
            , states)





