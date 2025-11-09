import gymnasium as gym
from skills import SkillEnv

env = SkillEnv(gym.make("MiniGrid-DoorKey-5x5-v0", render_mode="rgb_array"))
obs, info = env.reset()

print("get_key (0)")
obs, R, done, info = env.step(0)
print(info)

print("open_door (1)")
obs, R, done, info = env.step(1)
print(info)

print("go_to_goal (3)")
obs, R, done, info = env.step(3)
print(info)

