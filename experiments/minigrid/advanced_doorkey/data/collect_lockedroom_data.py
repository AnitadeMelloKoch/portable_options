from experiments.minigrid.utils import environment_builder
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
from experiments.minigrid.utils import actions 

from gymnasium.core import Env, Wrapper

class DataCollectorWrapper(Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)
    
    def _randomize_env(self):
        possible_player_pos = []
        for x in range(self.env.unwrapped.height - 5):
            for y in range(self.env.unwrapped.width - 5):
                cell = self.env.unwrapped.grid.get(x, y)
                if cell:
                    if cell.type == "door":
                        rand = np.random.randint(0, 100)
                        if rand < 30:
                            cell.is_locked = False
                            cell.is_open = True
                else:
                    possible_player_pos.append((x, y))
        
        new_player_pos = random.choice(possible_player_pos)
        self.env.unwrapped.agent_pos = new_player_pos
        self.env.unwrapped.agent_dir = np.random.randint(0, 4)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset()
        
        self._randomize_env()
        
        obs, _, _, info = self.env.step(actions.LEFT)
        obs, _, _, info = self.env.step(actions.RIGHT)
        
        return obs, info

if __name__ == "__main__":

    states = []

    for x in tqdm(range(50000)):
        env = environment_builder('MiniGrid-LockedRoom-v0',
                                seed=np.random.randint(0, 10000),
                                grayscale=False)
        env = DataCollectorWrapper(env)
        obs, _ = env.reset()
        
        states.append(obs)
        
        # fig, axes = plt.subplots()
        # axes.imshow(np.transpose(obs, axes=(1,2,0)))
        # plt.show()

    print(len(states))

    np.save("resources/minigrid_images/lockedroom_random_states.npy", states)

