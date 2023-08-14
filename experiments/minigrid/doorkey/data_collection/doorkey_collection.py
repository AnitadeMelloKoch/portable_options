from experiments.minigrid.utils import environment_builder, actions
from experiments.minigrid.doorkey.core.doorkey_option_wrapper import DoorKeyEnvOptionWrapper
import matplotlib.pyplot as plt 
import numpy as np
import random

init_positive_image = []
init_negative_image = []
term_positive_image = []
term_negative_image = []

fig = plt.figure(num=1, clear=True)
ax = fig.add_subplot()

def perform_action(env, 
                   action, 
                   steps, 
                   init_positive=None, 
                   term_positive=None):
    
    for _ in range(steps):
        
        state, _, terminated, _  = env.step(action)

        screen = env.render()
        ax.imshow(screen)
        plt.show(block=False)
        plt.pause(0.5)
        
        if init_positive is None:
            user_input = input("Initiation: (y) positive (n) negative")
            if user_input == "y":
                init_positive.append(state)
            elif user_input == "n":
                init_negative_image.append(state)
            else:
                print("Not saved to either")

        if term_positive is None:
            user_input = input("Termination: (y) positive (n) negative")
            if user_input == "y":
                term_positive.append(state)
            elif user_input == "n":
                term_negative_image.append(state)
            else:
                print("Not saved to either")
    
        if init_positive is True:
            print("in init set True")
            init_positive_image.append(state)
        elif init_positive is False:
            print("in init set False")
            init_negative_image.append(state)
        else:
            print("Not saved to either init")
        
        if term_positive is True:
            print("in term set True")
            term_positive_image.append(state)
        elif term_positive is False:
            print("in term set False")
            term_negative_image.append(state)
        else:
            print("Not saved to either term")
    
training_seed = 0

env = environment_builder('MiniGrid-DoorKey-8x8-v0', seed=training_seed, grayscale=False)
state, _ = env.reset()

screen = env.render()
ax.imshow(screen)
plt.show(block=False)

user_input = input("Initiation: (y) positive (n) negative ")
if user_input == "y":
    init_positive_image.append(state)
elif user_input == "n":
    init_negative_image.append(state)
else:
    print("Not saved to either")

user_input = input("Termination: (y) positive (n) negative ")
if user_input == "y":
    term_positive_image.append(state)
elif user_input == "n":
    term_negative_image.append(state)
else:
    print("Not saved to either")
    

perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.LEFT, 3,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.LEFT, 3,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.LEFT, 4,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.LEFT, 4,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 3,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.LEFT, 3,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.LEFT, 3,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False)
# in front of key
perform_action(env, actions.PICKUP, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.LEFT, 3,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 2,       init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.LEFT, 3,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.LEFT, 3,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.LEFT, 4,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.LEFT, 4,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 3,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.LEFT, 3,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.LEFT, 3,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 4,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 1,       init_positive=False, term_positive=False)
# in front of closed door
perform_action(env, actions.TOGGLE, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 1,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 2,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 1,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.LEFT, 1,       init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.LEFT, 3,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.LEFT, 3,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.LEFT, 4,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.LEFT, 4,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 3,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.LEFT, 3,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.LEFT, 3,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 4,     init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 1,       init_positive=False, term_positive=False)
# in front of open door
perform_action(env, actions.FORWARD, 1)





base_file_name = "doorkey_gogoal_{}_colour".format(training_seed)

if len(init_positive_image) > 0:
    np.save('resources/minigrid_images/{}_initiation_positive.npy'.format(base_file_name), init_positive_image)
if len(init_negative_image) > 0:
    np.save('resources/minigrid_images/{}_initiation_negative.npy'.format(base_file_name), init_positive_image)
if len(term_positive_image) > 0:
    np.save('resources/minigrid_images/{}_termination_positive.npy'.format(base_file_name), term_positive_image)
if len(term_negative_image) > 0:
    np.save('resources/minigrid_images/{}_termination_negative.npy'.format(base_file_name), term_positive_image)




