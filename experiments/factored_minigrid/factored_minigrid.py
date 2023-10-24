import gymnasium as gym
# import factored_minigrid
import matplotlib.pyplot as plt
from minigrid.wrappers import RGBImgObsWrapper
from experiments.factored_minigrid.utils import environment_builder
from experiments.factored_minigrid.doorkey.core.policy_train_wrapper import FactoredDoorKeyPolicyTrainWrapper

def check_option_complete(info):
    return False

env = FactoredDoorKeyPolicyTrainWrapper(environment_builder('FactoredMiniGrid-DoorKey-16x16-v0',
                        seed=0),
                        check_option_complete=check_option_complete,
                        key_picked_up=True,
                        door_open=True)

img, _ = env.reset()

fig = plt.figure(num=1, clear=True)
ax = fig.add_subplot()
screen = env.render()
ax.imshow(screen)
plt.show()

img, _ = env.reset()


fig = plt.figure(num=1, clear=True)
ax = fig.add_subplot()
screen = env.render()
ax.imshow(screen)
plt.show()

img, _ = env.reset()


fig = plt.figure(num=1, clear=True)
ax = fig.add_subplot()
screen = env.render()
ax.imshow(screen)
plt.show()

# fig = plt.figure(num=1, clear=True)
# ax = fig.add_subplot()
# ax.imshow(img[0, :,:])
# plt.show()

# fig = plt.figure(num=1, clear=True)
# ax = fig.add_subplot()
# ax.imshow(img[1, :,:])
# plt.show()

# fig = plt.figure(num=1, clear=True)
# ax = fig.add_subplot()
# ax.imshow(img[2, :,:])
# plt.show()

# fig = plt.figure(num=1, clear=True)
# ax = fig.add_subplot()
# ax.imshow(img[3, :,:])
# plt.show()

# fig = plt.figure(num=1, clear=True)
# ax = fig.add_subplot()
# ax.imshow(img[4, :,:])
# plt.show()

# fig = plt.figure(num=1, clear=True)
# ax = fig.add_subplot()
# ax.imshow(img[5, :,:])
# plt.show()




