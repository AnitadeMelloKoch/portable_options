import gym
import gym_treasure_game
from gym_treasure_game.envs.treasure_game import ObservationWrapper

import matplotlib.pyplot as plt

env = gym.make('treasure_game-v0')
env = ObservationWrapper(env)

obs = env.reset()

fig = plt.figure(clear=True)
ax = fig.add_subplot()
ax.imshow(obs)
plt.show()

