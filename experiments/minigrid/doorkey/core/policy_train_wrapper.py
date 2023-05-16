from gymnasium.core import Env, Wrapper 
from experiments.minigrid.utils import actions 
import numpy as np 
import matplotlib.pyplot as plt

class DoorKeyPolicyTrainWrapper(Wrapper):
    def __init__(self, 
                 env,
                 check_option_complete,
                 option_reward=1,
                 key_picked_up=False,
                 door_unlocked=False,
                 door_open=False):
        super().__init__(env)
        
        self.door_pos = ()
        self.key_pos = ()
        self.goal_pos = ()
        
        self._find_objs()
        
        self.check_option_complete = check_option_complete
        self.option_reward = option_reward
        
        self.key_picked_up = key_picked_up
        self.door_unlocked = door_unlocked
        self.door_open = door_open
    
    def _find_objs(self):
        print('looking for objects')
        for x in range(self.env.unwrapped.width):
            for y in range(self.env.unwrapped.height):
                cell = self.env.unwrapped.grid.get(x, y)
                if cell:
                    if cell.type == "key":
                        self.key_pos = (x, y)
                    elif cell.type == "door":
                        self.door_pos = (x, y)
                    elif cell.type == "goal":
                        self.goal_pos = (x, y)
                        
    def _modify_info_dict(self, info):
        info['key_pos'] = self.key_pos
        info['door_pos'] = self.door_pos
        info['goal_pos'] = self.goal_pos
        info['seed'] = self.env.env_seed
        
        return info
        
    def step(self, action):
        # print("action:", action)
        obs, reward, done, info = self.env.step(action)
        info = self._modify_info_dict(info)
        if self.check_option_complete(info):
            # fig = plt.figure(num=1, clear=True)
            # ax = fig.add_subplot()
            # screen = self.env.render()
            # ax.imshow(screen)
            # plt.show(block=False)
            # input("Option completed. Continue?")
            return obs, 1, True, info
        else:
            # fig = plt.figure(num=1, clear=True)
            # ax = fig.add_subplot()
            # screen = self.env.render()
            # ax.imshow(screen)
            # plt.show(block=False)
            # input("continue?")
            return obs, 0, done, info
        
    def reset(self):
        obs, info = self.env.reset()
        
        if self.key_picked_up or self.door_unlocked or self.door_open:
            key = self.env.unwrapped.grid.get(self.key_pos[0], self.key_pos[1])
            self.env.unwrapped.carrying = key
            self.env.unwrapped.carrying.cur_pos = np.array([-1, -1])
            self.env.unwrapped.grid.set(self.key_pos[0],
                                        self.key_pos[1],
                                        None)
        
            if self.door_unlocked or self.door_open:
                door = self.env.unwrapped.grid.get(self.door_pos[0], self.door_pos[1])
                door.is_locked = False
            
                if self.door_open:
                    door = self.env.unwrapped.grid.get(self.door_pos[0], self.door_pos[1])
                    door.is_locked = False
                    door.is_open = True
        
            obs, _, _, info = self.env.step(5)
        
        info = self._modify_info_dict(info)
        # print("player pos:", info["player_pos"])
        return obs, info