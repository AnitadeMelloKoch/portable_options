from experiments.minigrid.utils import environment_builder, actions
import matplotlib.pyplot as plt
import random
from experiments.minigrid.doorkey.core.doorkey_option_wrapper import DoorKeyEnvOptionWrapper

# 'MiniGrid-DoorKey-5x5-v0'     => seems to be 3x3
# 'MiniGrid-DoorKey-6x6-v0'     => seems to be 3x3
# 'MiniGrid-DoorKey-8x8-v0'
# 'MiniGrid-DoorKey-16x16-v0'

env = environment_builder('MiniGrid-DoorKey-16x16-v0', 
                          seed=random.randint(0,100),
                          grayscale=False) 
env = DoorKeyEnvOptionWrapper(env)

obs, info = env.reset()
obs = obs[0]
print(obs.shape)
print(env.unwrapped.agent_dir)
print(env.unwrapped.agent_pos)

fig = plt.figure(num=1, clear=True)
ax = fig.add_subplot()
screen = env.render()
ax.imshow(screen)
# ax.imshow(obs.squeeze())
plt.show()
fig = plt.figure(num=1, clear=True)
ax = fig.add_subplot()
screen = env.render()
ax.imshow(screen)
plt.show(block=False)
plt.pause(0.5)

env.step(7)
env.step(actions.PICKUP)
env.step(8)

# for _ in range(30):
#     action = random.randint(0, 6)
#     print(action)
#     fig = plt.figure(num=1, clear=True)
#     ax = fig.add_subplot()
#     screen = env.render()
#     ax.imshow(screen)
#     # ax.imshow(obs.squeeze())
#     plt.show(block=False)
#     plt.pause(0.5)
    
#     obs, _, _, _ = env.step(action)

