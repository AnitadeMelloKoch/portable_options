from experiments.minigrid.utils import environment_builder, actions
from experiments.minigrid.advanced_doorkey.core.policy_train_wrapper import AdvancedDoorKeyPolicyTrainWrapper
import matplotlib.pyplot as plt 
import numpy as np 

fig = plt.figure(num=1, clear=True)
ax = fig.add_subplot()

def perform_action(env, 
                   action, 
                   steps):
    
    for _ in range(steps):
        
        state, _, terminated, _  = env.step(action)

        screen = env.render()
        ax.imshow(np.transpose(state, (1,2,0)))
        plt.show(block=False)
        input("continue")
        

env = environment_builder('AdvancedDoorKey-8x8-v0', seed=2, grayscale=False)
env = AdvancedDoorKeyPolicyTrainWrapper(env,
                                        door_colour="red")

state, _ = env.reset()

screen = env.render()
ax.imshow(np.transpose(state, (1,2,0)))
plt.show(block=False)

input("continue")

perform_action(env, actions.LEFT, 2)
perform_action(env, actions.PICKUP, 1)
perform_action(env, actions.LEFT, 2)
perform_action(env, actions.FORWARD, 1)
perform_action(env, actions.PICKUP, 1)
perform_action(env, actions.LEFT, 2)
perform_action(env, actions.FORWARD, 1)
perform_action(env, actions.LEFT, 1)
perform_action(env, actions.FORWARD, 3)
print("toggling")
perform_action(env, actions.TOGGLE, 1)
perform_action(env, actions.FORWARD, 4)












