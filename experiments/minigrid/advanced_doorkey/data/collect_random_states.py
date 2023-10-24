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
# colours = ["red", "green", "blue", "purple", "yellow"]

# fig, ax = plt.subplots()

states = []

for x in range(4):
    for y in tqdm(range(2000)):
        for colour in colours:
            remaining_col = list(filter(lambda c: c != colour, colours))
            random.shuffle(remaining_col)
            seed = np.random.randint(low=0, high=1000)
            env = environment_builder('AdvancedDoorKey-8x8-v0',
                                      seed=seed,
                                      grayscale=False)
            env = AdvancedDoorKeyPolicyTrainWrapper(env,
                                                    key_collected=key[x],
                                                    door_unlocked=door_unlocked[x],
                                                    door_open=door_open[x],
                                                    door_colour=colour,
                                                    key_colours=remaining_col)
            
            obs, _ = env.reset()
            
            states.append(obs)

# fig = plt.figure(num=1, clear=True)
# ax = fig.add_subplot()

# for door_colour in colours:
#     states = []
#     for x in range(4):
#         for _ in tqdm(range(5)):
#             remaining_col = list(filter(lambda c: c!= door_colour, colours))
#             for train_colour in remaining_col:
#                 possible_key_colours = list(filter(lambda c: c!= train_colour, remaining_col))
#                 # other_col = random.choice(possible_key_colours)
#                 # key_cols = [train_colour, other_col]
#                 key_cols = random.sample(possible_key_colours, 2)
#                 random.shuffle(key_cols)
#                 env = environment_builder('AdvancedDoorKey-8x8-v0',
#                                       seed=0,
#                                       grayscale=False)
#                 env = AdvancedDoorKeyPolicyTrainWrapper(env,
#                                                         key_collected=key[x],
#                                                         door_unlocked=door_unlocked[x],
#                                                         door_open=door_open[x],
#                                                         door_colour=door_colour,
#                                                         key_colours=key_cols)
                
#                 seed = np.random.randint(low=0, high=20)
                
#                 obs, _ = env.reset(agent_reposition_attempts=seed)

#                 states.append(obs)

                # ax.imshow(np.transpose(obs, (1,2,0)))
                # plt.show(block=False)
                # input("key collected {}\ndoor unlocked {}\ndoor open {}".format(key[x],
                #                                                                 door_unlocked[x],
                #                                                                 door_open[x]))

print(states[0].shape)
print(len(states))

np.save("resources/minigrid_images/adv_doorkey_8x8_random_states.npy", states)





