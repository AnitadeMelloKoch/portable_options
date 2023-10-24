from gymnasium.core import Env, Wrapper
import numpy as np 
import matplotlib.pyplot as plt 
from experiments.factored_minigrid.utils import actions

def get_key(info):
    if info['carrying']:
        if info['carrying'] == 'key':
            return True
    return False

def open_door(info):
    return info['door_open']

def go_goal(info):
    return info['player_pos'] == info['goal_pos']

class FactoredDoorKeyPolicyTrainWrapper(Wrapper):
    def __init__(self, 
                 env,
                 check_option_complete,
                 option_reward=1,
                 key_picked_up=False,
                 door_open=False,
                 max_timesteps=100):
        super().__init__(env)
        
        # factored doorkey has a key, door, goal and a useless box and ball
        
        self.object_pos = {}
        self.important_objects = ["door", "key", "goal", "box", "ball"]
        
        
        self.check_option_complete = check_option_complete
        self.option_reward = option_reward
        
        self.start_key_collected = key_picked_up
        self.start_door_open = door_open
        
        self.max_timesteps = max_timesteps
        self._timestep = 0
        
    def _find_objs(self):
        # look for all objects in env
        for x in range(self.env.unwrapped.width):
            for y in range(self.env.unwrapped.height):
                cell = self.env.unwrapped.grid.get(x, y)
                if cell:
                    if cell.type in self.important_objects:
                        self.object_pos[cell.type] = (x,y)
    
    def _modify_info_dict(self, info):
        info['key_pos'] = self.object_pos["key"]
        info['door_pos'] = self.object_pos["door"]
        info['door_open'] = self.env.unwrapped.grid.get(self.object_pos["door"][0], self.object_pos["door"][1]).is_open
        info['goal_pos'] = self.object_pos["goal"]
        info['box_pos'] = self.object_pos["box"]
        info['ball_pos'] = self.object_pos["ball"]
        info['player_pos'] = tuple(self.env.agent_pos)
        info['carrying'] = self.env.unwrapped.carrying
        info['timestep'] = self._timestep
        if info['carrying']:
            info['carrying'] = info['carrying'].type
        
        return info
    
    def step(self, action):
        obs, _, done, info = self.env.step(action)
        self._timestep += 1
        info = self._modify_info_dict(info)
        if self.check_option_complete(info):
            return obs, 1, True, info
        if done:
            return obs, 0, done, info 
        if self._timestep >= self.max_timesteps:
            return obs, 0, True, info 
        return obs, 0, done, info 
    
    def reset(self):
        obs, info = self.env.reset()
        self._find_objs()
        
        
        if self.start_key_collected or self.start_door_open:
            key = self.env.unwrapped.grid.get(self.object_pos["key"][0], self.object_pos["key"][1])
            self.env.unwrapped.carrying = key
            self.env.unwrapped.carrying.cur_pos = np.array([-1, -1])
            self.env.unwrapped.grid.set(self.object_pos["key"][0], 
                                        self.object_pos["key"][1],
                                        None)
            if self.start_door_open:
                door = self.env.unwrapped.grid.get(self.object_pos["door"][0],
                                                   self.object_pos["door"][1])
                door.is_locked = False
                door.is_open = True
        
        new_agent_pos_found = False
        while not new_agent_pos_found:
            splitIdx = self.env.unwrapped.splitIdx
            height = self.env.unwrapped.height
            rand_x = np.random.randint(0, splitIdx)
            rand_y = np.random.randint(0, height)
            cell = self.env.unwrapped.grid.get(rand_x, rand_y)
            if cell is None:
                new_agent_pos_found = True
        
        self.env.unwrapped.agent_pos = (rand_x, rand_y)
        self.env.unwrapped.agent_dir = np.random.randint(0, 4)
        obs, _, _, _ = self.env.step(actions.RIGHT)
        obs, _, _, _ = self.env.step(actions.LEFT)
        
        self._timestep = 0
        
        info = self._modify_info_dict(info)
        
        return obs, info


