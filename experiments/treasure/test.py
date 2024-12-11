import gym
import gym_treasure_game
from gym_treasure_game.envs.treasure_game import ObservationWrapper

import matplotlib.pyplot as plt
from experiments.treasure.treasure_wrapper import TreasureInfoWrapper

env = gym.make('treasure_game-v0')
env = TreasureInfoWrapper(env)

obs = env.reset()

print(obs)
print(env.available_mask)

