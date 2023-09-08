import gymnasium as gym
from procgen import ProcgenEnv
import matplotlib.pyplot as plt 
import numpy as np
from tqdm import tqdm


# print(obs)

# fig, axes = plt.subplots()
# axes.imshow(obs)
# plt.show()

# 15
states = []
for y in tqdm(range(2000)):
    env = ProcgenEnv(num_envs=1,
                    env_name="coinrun",
                    num_levels=10,
                    center_agent=True)

    obs = env.reset()
    obs = obs["rgb"].squeeze()
    for x in range(50):
        obs, _, _, _ = env.step(np.array([np.random.randint(0, 10)]))
        
        obs = obs["rgb"].squeeze()
        
        states.append(obs)

print(len(states))
np.save("resources/procgen/coinrun.npy", states)
print("saved?")

