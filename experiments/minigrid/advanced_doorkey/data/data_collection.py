from experiments.minigrid.utils import environment_builder, actions, factored_environment_builder
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

        state = state.numpy()
        
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
training_seed = 4
colours = ["red", "green", "blue", "purple", "yellow", "grey"]
# door_colour = 'grey'
key_colour = 'red'
door_colour = 'blue'
#---------------------------------------------------------------------------------------------#
###############################################################################################
###############################################################################################
###############################################################################################

env = environment_builder('AdvancedDoorKey-8x8-v0', seed=training_seed, grayscale=False)
# env = AdvancedDoorKeyPolicyTrainWrapper(env,
#                                         door_colour=door_colour)
env = AdvancedDoorKeyPolicyTrainWrapper(env,
                                        door_colour=door_colour,
                                        key_colours=[key_colour,
                                                     "yellow",
                                                     "grey"])

# env = AdvancedDoorKeyPolicyTrainWrapper(
#     factored_environment_builder(
#         'AdvancedDoorKey-8x8-v0',
#         seed=training_seed
#     ),
#     door_colour=door_colour
# )

state, _ = env.reset()

state = state.numpy()

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
    

perform_action(env, actions.LEFT, 3                 , init_positive=True, term_positive=False, show=True)

#######################################################################################
#######################################################################################
##                                PICK UP KEY                                        ##
#######################################################################################
#######################################################################################


#######################################################################################
#######################################################################################
##                                UNLOCK DOOR                                        ##
#######################################################################################
#######################################################################################


#######################################################################################
#######################################################################################
##                                OPEN DOOR                                          ##
#######################################################################################
#######################################################################################



perform_action(env, actions.PICKUP, 1)
# 64

print(init_positive_image[0])


base_file_name = "adv_doorkey_8x8_get{}key_door{}_{}".format(key_colour, door_colour, training_seed)

if len(init_positive_image) > 0:
    np.save('resources/minigrid_factored/{}_initiation_positive.npy'.format(base_file_name), init_positive_image)
if len(init_negative_image) > 0:
    np.save('resources/minigrid_factored/{}_initiation_negative.npy'.format(base_file_name), init_negative_image)
if len(term_positive_image) > 0:
    np.save('resources/minigrid_factored/{}_termination_positive.npy'.format(base_file_name), term_positive_image)
if len(term_negative_image) > 0:
    np.save('resources/minigrid_factored/{}_termination_negative.npy'.format(base_file_name), term_negative_image)














