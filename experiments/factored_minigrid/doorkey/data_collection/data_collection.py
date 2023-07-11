from experiments.factored_minigrid.utils import environment_builder, actions
import matplotlib.pyplot as plt 
import numpy as np
import random

init_positive_image = []
init_positive_locs = []
init_negative_image = []
init_negative_locs = []
term_positive_image = []
term_positive_locs = []
term_negative_image = []
term_negative_locs = []

def perform_action(env, 
                   action, 
                   steps, 
                   init_positive=None,
                   term_positive=None):
    fig = plt.figure(num=1, clear=True)
    ax = fig.add_subplot()
    for _ in range(steps):
        
        state, _, _, info  = env.step(action)
        
        object_locs = info["object_locations"]
        screen = env.render()
        ax.imshow(screen)
        plt.show(block=False)
        plt.pause(0.4)
        
        if init_positive is None:
            user_input = input("Initiation: (y) positive (n) negative")
            if user_input == "y":
                init_positive = True
            elif user_input == "n":
                init_positive = False
                
        if term_positive is None:
            user_input = input("Termination: (y) positive (n) negative")
            if user_input == "y":
                term_positive = True
            elif user_input == "n":
                term_positive = False

        if init_positive is True:
            print("in init set True")
            init_positive_image.append(state)
            init_positive_locs.append(object_locs)
        elif init_positive is False:
            print("in init set False")
            init_negative_image.append(state)
            init_negative_locs.append(object_locs)
        else:
            print("Not saved to either init")
        
        if term_positive is True:
            print("in term set True")
            term_positive_image.append(state)
            term_positive_locs.append(object_locs)
        elif term_positive is False:
            print("in term set False")
            term_negative_image.append(state)
            term_negative_locs.append(object_locs)
        else:
            print("Not saved to either term")

training_seed = 2

env = environment_builder('FactoredMiniGrid-DoorKey-8x8-v0', seed=training_seed)
state, info = env.reset()
object_locs = info["object_locations"]



fig = plt.figure(num=1, clear=True)
ax = fig.add_subplot()
screen = env.render()
ax.imshow(screen)
plt.show(block=False)

user_input = input("Initiation: (y) positive (n) negative ")
if user_input == "y":
    init_positive_image.append(state)
    init_positive_locs.append(object_locs)
elif user_input == "n":
    init_negative_image.append(state)
    init_negative_locs.append(object_locs)
else:
    print("Not saved to either")

user_input = input("Termination: (y) positive (n) negative ")
if user_input == "y":
    term_positive_image.append(state)
    term_positive_locs.append(object_locs)
elif user_input == "n":
    term_negative_image.append(state)
    term_negative_locs.append(object_locs)
else:
    print("Not saved to either")

perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 6,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 4,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 4,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 6,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 2,      init_positive=False, term_positive=False)
perform_action(env, actions.LEFT, 3,         init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.LEFT, 4,         init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.LEFT, 4,         init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.LEFT, 3,         init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 4,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 4,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 4,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.LEFT, 3,         init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 4,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.LEFT, 3,         init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.LEFT, 3,         init_positive=False, term_positive=False)
# in front of key
perform_action(env, actions.PICKUP, 1,       init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 4,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 4,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 4,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.LEFT, 3,         init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.LEFT, 3,         init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 4,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 4,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 4,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 4,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.LEFT, 3,         init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 2,      init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 6,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 4,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 4,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 6,        init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 2,      init_positive=False, term_positive=False)
perform_action(env, actions.LEFT, 3,         init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.LEFT, 4,         init_positive=False, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=False)
perform_action(env, actions.RIGHT, 3,        init_positive=False, term_positive=False)
# in front of closed door
perform_action(env, actions.TOGGLE, 1,       init_positive=True, term_positive=False)
perform_action(env, actions.RIGHT, 1,        init_positive=True, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=True, term_positive=False)
perform_action(env, actions.LEFT, 3,         init_positive=True, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=True, term_positive=False)
perform_action(env, actions.RIGHT, 4,        init_positive=True, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=True, term_positive=False)
perform_action(env, actions.RIGHT, 4,        init_positive=True, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=True, term_positive=False)
perform_action(env, actions.RIGHT, 3,        init_positive=True, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=True, term_positive=False)
perform_action(env, actions.RIGHT, 3,        init_positive=True, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=True, term_positive=False)
perform_action(env, actions.RIGHT, 4,        init_positive=True, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=True, term_positive=False)
perform_action(env, actions.RIGHT, 4,        init_positive=True, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=True, term_positive=False)
perform_action(env, actions.LEFT, 3,         init_positive=True, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=True, term_positive=False)
perform_action(env, actions.LEFT, 3,         init_positive=True, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=True, term_positive=False)
perform_action(env, actions.RIGHT, 4,        init_positive=True, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=True, term_positive=False)
perform_action(env, actions.RIGHT, 4,        init_positive=True, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=True, term_positive=False)
perform_action(env, actions.LEFT, 3,         init_positive=True, term_positive=False)
perform_action(env, actions.FORWARD, 4,      init_positive=True, term_positive=False)
perform_action(env, actions.RIGHT, 6,        init_positive=True, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=True, term_positive=False)
perform_action(env, actions.RIGHT, 3,        init_positive=True, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=True, term_positive=False)
perform_action(env, actions.RIGHT, 4,        init_positive=True, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=True, term_positive=False)
perform_action(env, actions.RIGHT, 3,        init_positive=True, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=True, term_positive=False)
perform_action(env, actions.RIGHT, 4,        init_positive=True, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=True, term_positive=False)
perform_action(env, actions.RIGHT, 3,        init_positive=True, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=True, term_positive=False)
perform_action(env, actions.RIGHT, 6,        init_positive=True, term_positive=False)
perform_action(env, actions.FORWARD, 2,      init_positive=True, term_positive=False)
perform_action(env, actions.LEFT, 3,         init_positive=True, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=True, term_positive=False)
perform_action(env, actions.LEFT, 4,         init_positive=True, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=True, term_positive=False)
perform_action(env, actions.RIGHT, 3,        init_positive=True, term_positive=False)
# in front of open door
perform_action(env, actions.FORWARD, 1,      init_positive=True, term_positive=False)
perform_action(env, actions.RIGHT, 4,        init_positive=True, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=True, term_positive=False)
perform_action(env, actions.LEFT, 1,         init_positive=True, term_positive=False)
perform_action(env, actions.FORWARD, 2,      init_positive=True, term_positive=False)
perform_action(env, actions.RIGHT, 6,        init_positive=True, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=True, term_positive=False)
perform_action(env, actions.RIGHT, 4,        init_positive=True, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=True, term_positive=False)
perform_action(env, actions.RIGHT, 4,        init_positive=True, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=True, term_positive=False)
perform_action(env, actions.RIGHT, 4,        init_positive=True, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=True, term_positive=False)
perform_action(env, actions.RIGHT, 4,        init_positive=True, term_positive=False)
perform_action(env, actions.FORWARD, 1,      init_positive=False, term_positive=True)
perform_action(env, actions.RIGHT, 6,        init_positive=False, term_positive=True)



base_file_name = "doorkey_gogoal_{}".format(training_seed)

if len(init_positive_image) > 0:
    np.save('resources/minigrid_images/{}_initiation_image_positive.npy'.format(base_file_name), init_positive_image)
    np.save('resources/minigrid_images/{}_initiation_loc_positive.npy'.format(base_file_name), init_positive_locs)
if len(init_negative_image) > 0:
    np.save('resources/minigrid_images/{}_initiation_image_negative.npy'.format(base_file_name), init_negative_image)
    np.save('resources/minigrid_images/{}_initiation_loc_negative.npy'.format(base_file_name), init_negative_locs)
if len(term_positive_image) > 0:
    np.save('resources/minigrid_images/{}_termination_image_positive.npy'.format(base_file_name), term_positive_image)
    np.save('resources/minigrid_images/{}_termination_loc_positive.npy'.format(base_file_name), term_positive_locs)
if len(term_negative_image) > 0:
    np.save('resources/minigrid_images/{}_termination_image_negative.npy'.format(base_file_name), term_negative_image)
    np.save('resources/minigrid_images/{}_termination_loc_negative.npy'.format(base_file_name), term_negative_locs)
