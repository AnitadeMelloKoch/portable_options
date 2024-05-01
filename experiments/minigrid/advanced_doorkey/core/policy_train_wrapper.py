from typing import Tuple
from gymnasium.core import Env, Wrapper 
from minigrid.core.world_object import Key
from custom_minigrid.core.custom_world_object import CustomDoor, CustomKey
from experiments.minigrid.utils import actions 
import numpy as np 
import matplotlib.pyplot as plt 
from collections import namedtuple
import torch

KeyTuple = namedtuple("Key", "position colour")
DoorTuple = namedtuple("Door", "position colour")

class AdvancedDoorKeyPolicyTrainWrapper(Wrapper):
    def __init__(self, 
                 env: Env,
                 check_option_complete=lambda x: False,
                 option_reward: int=1,
                 key_colours: list=[],
                 door_colour: str=None,
                 key_collected: bool=False,
                 door_unlocked: bool=False,
                 door_open: bool=False,
                 time_limit: int=2000,
                 image_input: bool=True,
                 keep_colour: str="",
                 pickup_colour: str="",
                 force_door_closed=False,
                 force_door_open=False):
        super().__init__(env)
        
        self.objects = {
            "keys": [],
            "door": None,
            "goal": None
        }
        self.door_colour = door_colour
        self.key_colours = key_colours
        self.door_key_position = None
        self._timestep =  0
        self.time_limit = time_limit
        self.image_input = image_input
        self.keep_colour = keep_colour
        self.pickup_colour = pickup_colour
        
        self.door = None
        
        self.check_option_complete = check_option_complete
        self.option_reward = option_reward
        
        self.key_collected = key_collected
        self.door_unlocked = door_unlocked
        self.door_open = door_open
        self.force_door_closed = force_door_closed
        self.force_door_open = force_door_open
        
    
    def _modify_info_dict(self, info):
        info['timestep'] = self._timestep
        info['keys'] = self.objects["keys"]
        info['door'] = self.objects["door"]
        info['goal'] = self.objects["goal"]
        
        return info
    
    def _get_door_key(self):
        for key in self.objects['keys']:
            if key.colour == self.objects["door"].colour:
                return key
    
    def get_door_obj(self):
        door = self.env.unwrapped.grid.get(
            self.objects["door"].position[0], 
            self.objects["door"].position[1]
        )
        
        return door
    
    def reset(self, 
              agent_reposition_attempts=0,
              random_start=False,
              keep_colour="",
              pickup_colour="",
              force_door_closed=False,
              force_door_open=False,
              agent_position=None,
              collect_key=None,
              door_unlocked=None,
              door_open=None):
        
        obs, info = self.env.reset()
        
        self._find_objs()
        self._set_door_colour()
        self._set_key_colours()
        
        if collect_key is None:
            collect_key = self.key_collected
        
        if door_open is None:
            door_open = self.door_open
        
        if door_unlocked is None:
            door_unlocked = self.door_unlocked
        
        if collect_key:
            correct_key = self._get_door_key()
            key = self.env.unwrapped.grid.get(correct_key.position[0], correct_key.position[1])
            self.env.unwrapped.carrying.append(key)
            key.cur_pos = np.array([-1, -1])
            self.env.unwrapped.grid.set(correct_key.position[0],
                                        correct_key.position[1],
                                        None)
        
        if door_unlocked or door_open:
            door = self.env.unwrapped.grid.get(
                self.objects["door"].position[0], 
                self.objects["door"].position[1]
            )
            door.is_locked = False
        
            if door_open:
                door = self.env.unwrapped.grid.get(
                    self.objects["door"].position[0], 
                    self.objects["door"].position[1]
                )
                door.is_locked = False
                door.is_open = True
        
        if random_start is True:
            self.random_start(keep_colour=keep_colour,
                              pickup_colour=pickup_colour,
                              force_door_closed=force_door_closed,
                              force_door_open=force_door_open)
        
        obj = None
        
        if agent_position is not None:
            agent_x, agent_y = agent_position
            obj = self.env.unwrapped.grid.get(agent_x, agent_y)
        
        # if agent_position is None or obj is not None:
        if agent_position is None:
            self.env.unwrapped.place_agent_randomly(agent_reposition_attempts)
        else:
            self.env.unwrapped.agent_pos = agent_position
        
        self._find_objs()
        
        obs, _, _, info = self.env.step(actions.LEFT)
        obs, _, _, info = self.env.step(actions.RIGHT)
        
        self.env.unwrapped.time_step = 0
        self._timestep = 0
        
        info = self._modify_info_dict(info)
        
        # fig = plt.figure(num=1, clear=True)
        # ax = fig.add_subplot()
        # ax.imshow(np.transpose(obs, axes=[1,2,0]))
        # plt.show(block=False)
        # input("Option completed. Continue?")
        
        if type(obs) is np.ndarray:
            obs = torch.from_numpy(obs).float()
        
        return obs, info
    
    def random_start(self, 
                     keep_colour="",
                     pickup_colour="",
                     force_door_closed=False,
                     force_door_open=False):
        # randomly move+pick up keys in the environment
        # randomly open/unlock door
        
        split_idx = self.env.unwrapped.splitIdx
        
        if keep_colour == "":
            keep_colour = self.keep_colour
        if pickup_colour == "":
            pickup_colour = self.pickup_colour
        if force_door_closed is False:
            force_door_closed = self.force_door_closed
        if force_door_open is False:
            force_door_open = self.force_door_open
        
        all_pos = [key.position for key in self.objects["keys"]]
        
        for key in self.objects["keys"]:
            if key.colour == keep_colour:
                continue
            
            pos = key.position
            all_pos.remove(pos)
            
            randval = np.random.rand()
            if (randval < 0.1) or (key.colour == pickup_colour):
                key_obj = self.env.unwrapped.grid.get(pos[0], pos[1])
                self.env.unwrapped.carrying.append(key_obj)
                key_obj.cur_pos = np.array([-1, -1])
                self.env.unwrapped.grid.set(pos[0], pos[1], None)
                
            else:
                new_pos_found = False
                while new_pos_found is False:
                    new_pos = (np.random.randint(1, split_idx), np.random.randint(1,7))
                    if not new_pos in all_pos:
                        new_pos_found = True
                
                key_obj = self.env.unwrapped.grid.get(pos[0], pos[1])
                # if key_obj is None:
                #     continue
                self.env.unwrapped.grid.set(pos[0], pos[1], None)
                key_obj.cur_pos = np.array([new_pos[0], new_pos[1]])
                self.env.unwrapped.grid.set(new_pos[0], new_pos[1], key_obj)
                all_pos.append(new_pos)
        
        if force_door_closed is False:
            randval = np.random.rand()
            if randval < 0.6 or force_door_open:
                door = self.env.unwrapped.grid.get(
                    self.objects["door"].position[0],
                    self.objects["door"].position[1],
                )
                door.is_locked = False
                if randval < 0.3 or force_door_open:
                    door.is_open = True
        
    
    def _find_objs(self):
        self.objects = {
            "keys": [],
            "door": None,
            "goal": None
        }
        for x in range(self.env.unwrapped.width):
            for y in range(self.env.unwrapped.height):
                cell = self.env.unwrapped.grid.get(x, y)
                if cell:
                    if cell.type == "key":
                        self.objects["keys"].append(
                            KeyTuple((x, y), cell.color)
                        )
                    elif cell.type == "door":
                        self.door = cell
                        self.objects["door"] = DoorTuple((x, y), cell.color)
                    elif cell.type == "goal":
                        self.objects["goal"] = (x, y)
                    elif cell.type == "wall":
                        continue
                    else:
                        raise Exception("Unrecognized object {} found at ({},{})".format(cell, x, y))
        
        if self.door_key_position is None:
            door_key = self._get_door_key()
            self.door_key_position = door_key.position
        
    def _set_door_colour(self):
        if self.door_colour is None:
            self.door_colour = self.objects["door"].colour
            return
        
        new_door = CustomDoor(self.door_colour, is_locked=True)
        
        self.env.unwrapped.grid.set(
            self.objects["door"].position[0],
            self.objects["door"].position[1],
            new_door
        )
        old_colour = self.objects["door"].colour
        self.objects["door"] = DoorTuple(self.objects["door"].position,
                                                self.door_colour)
        
        new_key = CustomKey(self.door_colour)
        keys = []
        
        for key in self.objects["keys"]:
            if key.colour == old_colour:
                self.env.unwrapped.grid.set(
                    key.position[0],
                    key.position[1],
                    new_key
                )
                keys.append(KeyTuple(key.position,
                                     self.door_colour))
            elif key.colour == self.door_colour:
                replace_key = CustomKey(old_colour)
                self.env.unwrapped.grid.set(
                    key.position[0],
                    key.position[1],
                    replace_key
                )
                keys.append(KeyTuple(key.position,
                                     old_colour))
            else:
                keys.append(key)
        self.objects["keys"] = keys
        
    def _set_key_colours(self):
        if len(self.key_colours) == 0:
            return
        
        c_idx = 0
        for idx, key in enumerate(self.objects["keys"]):
            if key.position == self.door_key_position:
                continue
            if c_idx >= len(self.key_colours):
                return
            
            colour = self.key_colours[c_idx]
            
            new_key = CustomKey(colour)
            self.env.unwrapped.grid.set(
                key.position[0],
                key.position[1],
                new_key
            )
            self.objects["keys"][idx] = KeyTuple(key.position,
                                                 colour)
            c_idx += 1
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._timestep += 1
        info = self._modify_info_dict(info)
        if self._timestep >= self.time_limit:
            done = True
        if self.image_input:
            if np.max(obs) > 1:
                obs = obs/255
        if type(obs) is np.ndarray:
            obs = torch.from_numpy(obs).float()
        if self.check_option_complete(self):
            # fig = plt.figure(num=1, clear=True)
            # ax = fig.add_subplot()
            # screen = self.env.render()
            # ax.imshow(screen)
            # plt.show(block=False)
            # input("Option completed. Continue?")
            return obs, 1, True, info
        else:
            # print(obs.shape)
            # print(obs)
            # fig = plt.figure(num=1, clear=True)
            # ax = fig.add_subplot()
            # # screen = self.env.render()
            # screen = np.transpose(obs.numpy(), axes=(1,2,0))
            # ax.imshow(screen)
            # plt.show(block=False)
            # input("continue?")
            return obs, 0, done, info

