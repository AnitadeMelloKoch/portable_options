import gymnasium as gym
from procgen import ProcgenEnv
import matplotlib.pyplot as plt 
import numpy as np
from tqdm import tqdm



env_name = "coinrun"

# 15
states = []
for y in tqdm(range(2000)):
    env = ProcgenEnv(num_envs=1,
                    env_name=env_name,
                    num_levels=10,
                    center_agent=True,
                    distribution_mode="easy")

    obs = env.reset()
    obs = obs["rgb"].squeeze()
    
    # print(obs)
    
    # fig, axes = plt.subplots()
    # axes.imshow(obs)
    # plt.show()
    
    for x in range(50):
        obs, _, _, _ = env.step(np.array([np.random.randint(0, 10)]))
        
        obs = obs["rgb"].squeeze()
        
        states.append(obs)

print(len(states))
print(states[0].shape)
np.save("resources/procgen/easy_{}.npy".format(env_name), states)
print("saved?")

