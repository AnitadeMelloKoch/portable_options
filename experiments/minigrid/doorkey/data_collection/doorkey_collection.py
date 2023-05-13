from experiments.minigrid.utils import environment_builder, actions
import matplotlib.pyplot as plt 
import numpy as np
import random

def perform_action(env, 
                   action, 
                   steps, 
                   init_positive, 
                   init_negative,
                   term_positive,
                   term_negative):
    fig = plt.figure(num=1, clear=True)
    ax = fig.add_subplot()
    for _ in range(steps):
        
        state, _, terminated, _  = env.step(action)
        print(terminated)

        screen = env.render()
        ax.imshow(screen)
        plt.show(block=False)
        
        user_input = input("Initiation: (y) positive (n) negative")
        if user_input == "y":
            init_positive.append(state)
        elif user_input == "n":
            init_negative.append(state)
        else:
            print("Not saved to either")

        user_input = input("Termination: (y) positive (n) negative")
        if user_input == "y":
            term_positive.append(state)
        elif user_input == "n":
            term_negative.append(state)
        else:
            print("Not saved to either")
    
    return init_positive, init_negative, term_positive, term_negative

training_seed = 0

env = environment_builder('MiniGrid-DoorKey-8x8-v0', seed=training_seed)
state, _ = env.reset()

init_positive = []
init_negative = []
term_positive = []
term_negative = []

fig = plt.figure(num=1, clear=True)
ax = fig.add_subplot()
screen = env.render()
ax.imshow(screen)
plt.show(block=False)

user_input = input("Initiation: (y) positive (n) negative")
if user_input == "y":
    init_positive.append(state)
elif user_input == "n":
    init_negative.append(state)
else:
    print("Not saved to either")

user_input = input("Termination: (y) positive (n) negative")
if user_input == "y":
    term_positive.append(state)
elif user_input == "n":
    term_negative.append(state)
else:
    print("Not saved to either")
    

# possible_actions = [actions.LEFT, actions.FORWARD, actions.RIGHT]
# max_len = 3

# for _ in range(40):
#     action = random.choice(possible_actions)
#     step_num = random.choice(np.arange(max_len))
#     init_positive, init_negative, term_positive, term_negative = perform_action(env, action, step_num, init_positive, init_negative, term_positive, term_negative)

# base_file_name = "doorkey_getkey"

# if len(init_positive > 0):
#     np.save('{}_initiation_positive.npy'.format(base_file_name), init_positive)
# if len(init_negative > 0):
#     np.save('{}_initiation_negative.npy'.format(base_file_name), init_positive)
# if len(term_positive > 0):
#     np.save('{}_termination_positive.npy'.format(base_file_name), term_positive)
# if len(term_negative > 0):
#     np.save('{}_termination_negative.npy'.format(base_file_name), term_positive)




