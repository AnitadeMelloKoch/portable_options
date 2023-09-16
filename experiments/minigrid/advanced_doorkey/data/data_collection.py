from experiments.minigrid.utils import environment_builder, actions
from experiments.minigrid.advanced_doorkey.core.policy_train_wrapper import AdvancedDoorKeyPolicyTrainWrapper
import matplotlib.pyplot as plt 
import numpy as np 

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
                   term_positive=None,
                   show=True):
    
    for _ in range(steps):
        
        state, _, terminated, _  = env.step(action)

        if show:
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

###############################################################################################
###############################################################################################
###############################################################################################
#---------------------------------------------------------------------------------------------#
training_seed = 0
door_colour = 'blue'
key_colour = 'grey'
#---------------------------------------------------------------------------------------------#
###############################################################################################
###############################################################################################
###############################################################################################

env = environment_builder('AdvancedDoorKey-16x16-v0', seed=training_seed, grayscale=False)
env = AdvancedDoorKeyPolicyTrainWrapper(env,
                                        door_colour=door_colour)
# env = AdvancedDoorKeyPolicyTrainWrapper(env,
#                                         door_colour=door_colour,
#                                         key_colours=[key_colour,
#                                                      "red",
#                                                      "grey"])
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
    

perform_action(env, actions.FORWARD, 7,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.LEFT, 1,            init_positive=False, term_positive=False, show=False)
perform_action(env, actions.RIGHT, 2,           init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 6,         init_positive=False, term_positive=False, show=False)

perform_action(env, actions.RIGHT, 2,           init_positive=False, term_positive=False, show=False)
for _ in range(4):
    for _ in range(10):
        perform_action(env, actions.RIGHT, 4,   init_positive=False, term_positive=False, show=False)
        perform_action(env, actions.FORWARD, 1, init_positive=False, term_positive=False, show=False)

    perform_action(env, actions.LEFT, 3,        init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.RIGHT, 1,       init_positive=False, term_positive=False, show=False)

    for _ in range(10):
        perform_action(env, actions.RIGHT, 4,   init_positive=False, term_positive=False, show=False)
        perform_action(env, actions.FORWARD, 1, init_positive=False, term_positive=False, show=False)

    perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.LEFT, 1,        init_positive=False, term_positive=False, show=False)


for _ in range(4):
    perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)

perform_action(env, actions.RIGHT, 4,           init_positive=False, term_positive=False, show=False)

# in front of grey key
perform_action(env, actions.RIGHT, 1,           init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 1,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.LEFT, 1,            init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 2,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.LEFT, 1,            init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 1,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.RIGHT, 1,           init_positive=False, term_positive=False, show=False)
for _ in range(4):
    perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)
perform_action(env, actions.LEFT, 3,            init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 1,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.RIGHT, 1,           init_positive=False, term_positive=False, show=False)
for _ in range(10):
    perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)
perform_action(env, actions.RIGHT, 3,           init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 1,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.LEFT, 1,            init_positive=False, term_positive=False, show=False)

for _ in range(10):
    perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)

perform_action(env, actions.LEFT, 3,            init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 1,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.RIGHT, 1,           init_positive=False, term_positive=False, show=False)

for _ in range(10):
    perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)

perform_action(env, actions.RIGHT, 3,           init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 1,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.LEFT, 3,            init_positive=False, term_positive=False, show=False)

# in front of red key
perform_action(env, actions.LEFT, 1,            init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 1,         init_positive=False, term_positive=False, show=False)

perform_action(env, actions.RIGHT, 3,           init_positive=False, term_positive=False, show=False)

for _ in range(4):
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False, show=False)

# in front of yellow key
perform_action(env, actions.LEFT, 1,            init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 1,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.RIGHT, 1,           init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 2,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.RIGHT, 1,           init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 1,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.LEFT, 1,            init_positive=False, term_positive=False, show=False)

for _ in range(4):
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False, show=False)

# going to get red key
perform_action(env, actions.LEFT, 1,            init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 1,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.LEFT, 1,            init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 8,         init_positive=False, term_positive=False, show=False)

#########################################################################################################
#########################################################################################################
#########################################################################################################
#### Going to pick up red key
#########################################################################################################
#########################################################################################################
#########################################################################################################

perform_action(env, actions.PICKUP, 1,          init_positive=False, term_positive=False, show=True)
perform_action(env, actions.RIGHT, 1,           init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 12,        init_positive=False, term_positive=False, show=False)
perform_action(env, actions.LEFT, 1,            init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 2,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.RIGHT, 2,           init_positive=False, term_positive=False, show=False)

for _ in range(4):
    for _ in range(10):
        perform_action(env, actions.RIGHT, 4,   init_positive=False, term_positive=False, show=False)
        perform_action(env, actions.FORWARD, 1, init_positive=False, term_positive=False, show=False)

    perform_action(env, actions.LEFT, 3,        init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.RIGHT, 1,       init_positive=False, term_positive=False, show=False)

    for _ in range(10):
        perform_action(env, actions.RIGHT, 4,   init_positive=False, term_positive=False, show=False)
        perform_action(env, actions.FORWARD, 1, init_positive=False, term_positive=False, show=False)

    perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.LEFT, 1,        init_positive=False, term_positive=False, show=False)

for _ in range(4):
    perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)

perform_action(env, actions.RIGHT, 4,           init_positive=False, term_positive=False, show=False)
# in front of grey key
perform_action(env, actions.RIGHT, 1,           init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 1,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.LEFT, 1,            init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 2,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.LEFT, 1,            init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 1,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.RIGHT, 1,           init_positive=False, term_positive=False, show=False)
for _ in range(4):
    perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)

for _ in range(2):
    perform_action(env, actions.LEFT, 3,        init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.RIGHT, 1,       init_positive=False, term_positive=False, show=False)
    for _ in range(10):
        perform_action(env, actions.RIGHT, 4,   init_positive=False, term_positive=False, show=False)
        perform_action(env, actions.FORWARD, 1, init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.LEFT, 1,        init_positive=False, term_positive=False, show=False)

    for _ in range(10):
        perform_action(env, actions.RIGHT, 4,   init_positive=False, term_positive=False, show=False)
        perform_action(env, actions.FORWARD, 1, init_positive=False, term_positive=False, show=False)

perform_action(env, actions.RIGHT, 4,           init_positive=False, term_positive=False, show=False)
perform_action(env, actions.RIGHT, 1,           init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 1,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.LEFT, 3,            init_positive=False, term_positive=False, show=False)

for _ in range(4):
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False, show=False)

perform_action(env, actions.RIGHT, 1,           init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 1,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.LEFT, 1,            init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 2,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.LEFT, 1,            init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 1,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.RIGHT, 1,           init_positive=False, term_positive=False, show=False)
perform_action(env, actions.RIGHT, 4,           init_positive=False, term_positive=False, show=False)

for _ in range(4):
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False, show=False)

perform_action(env, actions.RIGHT, 1,           init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 9,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.RIGHT, 1,           init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 10,        init_positive=False, term_positive=False, show=False)

#########################################################################################################
#########################################################################################################
#########################################################################################################
# about to open door
#########################################################################################################
#########################################################################################################
#########################################################################################################

perform_action(env, actions.TOGGLE, 1,          init_positive=False, term_positive=False, show=True)
perform_action(env, actions.LEFT, 1,            init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 4,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.LEFT, 1,            init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 10,        init_positive=False, term_positive=False, show=False)

perform_action(env, actions.RIGHT, 2,           init_positive=False, term_positive=False, show=False)

for _ in range(4):
    for _ in range(10):
        perform_action(env, actions.RIGHT, 4,   init_positive=False, term_positive=False, show=False)
        perform_action(env, actions.FORWARD, 1, init_positive=False, term_positive=False, show=False)

    perform_action(env, actions.LEFT, 3,        init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.RIGHT, 1,       init_positive=False, term_positive=False, show=False)

    for _ in range(10):
        perform_action(env, actions.RIGHT, 4,   init_positive=False, term_positive=False, show=False)
        perform_action(env, actions.FORWARD, 1, init_positive=False, term_positive=False, show=False)

    perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.LEFT, 1,        init_positive=False, term_positive=False, show=False)

for _ in range(4):
    perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)

perform_action(env, actions.RIGHT, 4,           init_positive=False, term_positive=False, show=False)
# in front of grey key
perform_action(env, actions.RIGHT, 1,           init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 1,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.LEFT, 1,            init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 2,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.LEFT, 1,            init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 1,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.RIGHT, 1,           init_positive=False, term_positive=False, show=False)
for _ in range(4):
    perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)

for _ in range(2):
    perform_action(env, actions.LEFT, 3,        init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.RIGHT, 1,       init_positive=False, term_positive=False, show=False)
    for _ in range(10):
        perform_action(env, actions.RIGHT, 4,   init_positive=False, term_positive=False, show=False)
        perform_action(env, actions.FORWARD, 1, init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.LEFT, 1,        init_positive=False, term_positive=False, show=False)

    for _ in range(10):
        perform_action(env, actions.RIGHT, 4,   init_positive=False, term_positive=False, show=False)
        perform_action(env, actions.FORWARD, 1, init_positive=False, term_positive=False, show=False)

perform_action(env, actions.RIGHT, 4,           init_positive=False, term_positive=False, show=False)
perform_action(env, actions.RIGHT, 1,           init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 1,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.LEFT, 3,            init_positive=False, term_positive=False, show=False)

for _ in range(4):
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False, show=False)

perform_action(env, actions.RIGHT, 1,           init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 1,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.LEFT, 1,            init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 2,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.LEFT, 1,            init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 1,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.RIGHT, 1,           init_positive=False, term_positive=False, show=False)
perform_action(env, actions.RIGHT, 4,           init_positive=False, term_positive=False, show=False)

for _ in range(4):
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False, show=False)

perform_action(env, actions.RIGHT, 1,           init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 9,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.RIGHT, 1,           init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 10,        init_positive=False, term_positive=False, show=False)
#########################################################################################################
#########################################################################################################
#########################################################################################################
# about to close unlocked door
#########################################################################################################
#########################################################################################################
#########################################################################################################

perform_action(env, actions.TOGGLE, 1,          init_positive=False, term_positive=False, show=True)
perform_action(env, actions.LEFT, 1,            init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 4,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.LEFT, 1,            init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 10,        init_positive=False, term_positive=False, show=False)

perform_action(env, actions.RIGHT, 2,           init_positive=False, term_positive=False, show=False)

for _ in range(4):
    for _ in range(10):
        perform_action(env, actions.RIGHT, 4,   init_positive=False, term_positive=False, show=False)
        perform_action(env, actions.FORWARD, 1, init_positive=False, term_positive=False, show=False)

    perform_action(env, actions.LEFT, 3,        init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.RIGHT, 1,       init_positive=False, term_positive=False, show=False)

    for _ in range(10):
        perform_action(env, actions.RIGHT, 4,   init_positive=False, term_positive=False, show=False)
        perform_action(env, actions.FORWARD, 1, init_positive=False, term_positive=False, show=False)

    perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.LEFT, 1,        init_positive=False, term_positive=False, show=False)

for _ in range(4):
    perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)

perform_action(env, actions.RIGHT, 4,           init_positive=False, term_positive=False, show=False)
# in front of grey key
perform_action(env, actions.RIGHT, 1,           init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 1,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.LEFT, 1,            init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 2,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.LEFT, 1,            init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 1,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.RIGHT, 1,           init_positive=False, term_positive=False, show=False)
for _ in range(4):
    perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)

for _ in range(2):
    perform_action(env, actions.LEFT, 3,        init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.RIGHT, 1,       init_positive=False, term_positive=False, show=False)
    for _ in range(10):
        perform_action(env, actions.RIGHT, 4,   init_positive=False, term_positive=False, show=False)
        perform_action(env, actions.FORWARD, 1, init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.LEFT, 1,        init_positive=False, term_positive=False, show=False)

    for _ in range(10):
        perform_action(env, actions.RIGHT, 4,   init_positive=False, term_positive=False, show=False)
        perform_action(env, actions.FORWARD, 1, init_positive=False, term_positive=False, show=False)

perform_action(env, actions.RIGHT, 4,           init_positive=False, term_positive=False, show=False)
perform_action(env, actions.RIGHT, 1,           init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 1,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.LEFT, 3,            init_positive=False, term_positive=False, show=False)

for _ in range(4):
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False, show=False)

perform_action(env, actions.RIGHT, 1,           init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 1,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.LEFT, 1,            init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 2,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.LEFT, 1,            init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 1,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.RIGHT, 1,           init_positive=False, term_positive=False, show=False)
perform_action(env, actions.RIGHT, 4,           init_positive=False, term_positive=False, show=False)

for _ in range(4):
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.RIGHT, 4,       init_positive=False, term_positive=False, show=False)

perform_action(env, actions.RIGHT, 1,           init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 9,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.RIGHT, 1,           init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 10,        init_positive=False, term_positive=False, show=False)
#########################################################################################################
#########################################################################################################
#########################################################################################################
# about to open unlocked door
#########################################################################################################
#########################################################################################################
#########################################################################################################

perform_action(env, actions.TOGGLE, 1,          init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 1,         init_positive=False, term_positive=False, show=False)

perform_action(env, actions.RIGHT, 4,           init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 1,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.RIGHT, 4,           init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 1,         init_positive=False, term_positive=False, show=False)

perform_action(env, actions.LEFT, 1,            init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 4,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.LEFT, 3,            init_positive=False, term_positive=False, show=False)

for _ in range(6):
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.LEFT, 3,        init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.LEFT, 3,        init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.FORWARD, 1,     init_positive=False, term_positive=False, show=False)
    perform_action(env, actions.RIGHT, 3,       init_positive=False, term_positive=False, show=False)

perform_action(env, actions.FORWARD, 1,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.RIGHT, 6,           init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 1,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.LEFT, 1,            init_positive=False, term_positive=False, show=False)
perform_action(env, actions.FORWARD, 1,         init_positive=False, term_positive=False, show=False)
perform_action(env, actions.RIGHT, 3,           init_positive=False, term_positive=False, show=False)
# on goal
perform_action(env, actions.FORWARD, 1,         init_positive=False, term_positive=False, show=True)
perform_action(env, actions.RIGHT, 8,           init_positive=False, term_positive=False, show=False)


# perform_action(env, actions.PICKUP, 1)



base_file_name = "adv_doorkey_open{}door_door{}_{}".format(key_colour, door_colour, training_seed)

if len(init_positive_image) > 0:
    np.save('resources/minigrid_images/{}_initiation_positive.npy'.format(base_file_name), init_positive_image)
if len(init_negative_image) > 0:
    np.save('resources/minigrid_images/{}_initiation_negative.npy'.format(base_file_name), init_negative_image)
if len(term_positive_image) > 0:
    np.save('resources/minigrid_images/{}_termination_positive.npy'.format(base_file_name), term_positive_image)
if len(term_negative_image) > 0:
    np.save('resources/minigrid_images/{}_termination_negative.npy'.format(base_file_name), term_negative_image)













