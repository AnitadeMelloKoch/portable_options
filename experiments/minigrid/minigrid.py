from experiments.minigrid.utils import environment_builder, actions
import matplotlib.pyplot as plt
import random
from experiments.minigrid.doorkey.core.doorkey_option_wrapper import DoorKeyEnvOptionWrapper
from experiments.minigrid.doorkey.core.policy_train_wrapper import DoorKeyPolicyTrainWrapper

# 'MiniGrid-DoorKey-5x5-v0'     => seems to be 3x3
# 'MiniGrid-DoorKey-6x6-v0'     => seems to be 3x3
# 'MiniGrid-DoorKey-8x8-v0'
# 'MiniGrid-DoorKey-16x16-v0'

start_pos = [
    (1,1), (2,1), (3,1), (4,1),
    (1,2), (2,2), (3,2), (4,2),
    (1,3), (2,3), (3,3), (4,3),
    (1,4), (2,4), (3,4), 
    (1,5), (2,5),  
]

def perform_action(env, 
                   action, 
                   steps):
    for _ in range(steps):
        fig = plt.figure(num=1, clear=True)
        ax = fig.add_subplot()
        state, reward, terminated, _  = env.step(action)

        screen = env.render()
        ax.imshow(screen)
        plt.show()
        
        print(terminated)
        print(reward)

def at_key(info):
    obj_x, obj_y = info["key_pos"]
    player_x = info["player_x"]
    player_y = info["player_y"]
    
    if (abs(obj_x-player_x) + abs(obj_y-player_y)) <= 1:
        return True
    return False

env = environment_builder('MiniGrid-DoorKey-8x8-v0', 
                          seed=0,
                          grayscale=False,
                          random_reset=True,
                          random_starts=start_pos) 
env = DoorKeyPolicyTrainWrapper(env, at_key,
                                door_open=True)

for x in range(10):
    
    obs, info = env.reset()

    fig = plt.figure(num=1, clear=True)
    ax = fig.add_subplot()
    screen = env.render()
    ax.imshow(screen)
    plt.show(block=False)
    input("continue?")
    
    perform_action(env, actions.FORWARD, 2)



