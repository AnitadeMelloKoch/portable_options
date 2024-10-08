from typing import Tuple
from gymnasium.core import Env, Wrapper 
# from minigrid.core.world_object import Key, Door
from custom_minigrid.core.custom_world_object import CustomDoor, CustomKey
from experiments.minigrid.utils import actions 
import numpy as np 
import matplotlib.pyplot as plt 
import torch
from copy import deepcopy, copy
from PIL import Image


class KeyTuple():
    def __init__(self, position, colour) -> None:
        self.position = position
        self.colour = colour

class DoorTuple():
    def __init__(self, position, colour) -> None:
        self.position = position
        self.colour = colour

def default_check_option_complete(x):
    return False

class AdvancedDoorKeyPolicyTrainWrapper(Wrapper):
    def __init__(self, 
                 env: Env,
                 check_option_complete=default_check_option_complete,
                 state_size=(84,84),
                 term_size=(84,84),
                 option_reward: int=1,
                 key_colours: list=[],
                 door_colour: str=None,
                 key_collected: bool=False,
                 door_unlocked: bool=False,
                 door_open: bool=False,
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
        self.image_input = image_input
        self.keep_colour = keep_colour
        self.pickup_colour = pickup_colour
        # self.state_size = state_size
        # self.term_size = term_size
        
        self.door = None
        
        self.check_option_complete = check_option_complete
        self.option_reward = option_reward
        
        self.key_collected = key_collected
        self.door_unlocked = door_unlocked
        self.door_open = door_open
        self.force_door_closed = force_door_closed
        self.force_door_open = force_door_open
        
        self.obs = None
        
    
    def _modify_info_dict(self, info):
        info['timestep'] = self._timestep
        info['keys'] = self.objects["keys"]
        info['door'] = self.objects["door"]
        info['goal'] = self.objects["goal"]
        info['agent_pos'] = self.get_current_position()
        
        return info
    
    def _get_door_key(self):
        for key in self.objects['keys']:
            if key.colour == self.objects["door"].colour:
                return key
    
    def get_current_position(self):
        return self.env.unwrapped.agent_pos
    
    def print_env_objects(self):
        self._find_objs()
        for key in self.objects.keys():
            print(key)
            if key == "keys":
                for obj in self.objects[key]:
                    print(obj.position, obj.colour)
            elif key =="door":
                print(self.objects[key].position, self.objects[key].colour)
    
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
        
        self.obs = deepcopy(obs)
        
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
        
        obs, _, _, info = self.step(actions.LEFT)
        obs, _, _, info = self.step(actions.RIGHT)
                
        self.env.unwrapped.time_step = 0
        self._timestep = 0
        
        info = self._modify_info_dict(info)
        
        # fig = plt.figure(num=1, clear=True)
        # ax = fig.add_subplot()
        # ax.imshow(np.transpose(obs, axes=[1,2,0]))
        # plt.show(block=False)
        # input("Option completed. Continue?")
        
        
        return obs, info
    
    def get_term_state(self):
        obs = self.obs
        if type(obs) is not np.ndarray:
            obs = obs.numpy()
        if self.term_size is not None:
            img = Image.fromarray(obs)
            obs = np.asarray(img.resize(self.term_size, Image.BICUBIC))
        obs = torch.from_numpy(obs)
        obs = obs.permute((2,0,1))
        return obs.unsqueeze(0).float()
    
    def random_start(self, 
                     keep_colour="",
                     pickup_colour="",
                     force_door_closed=False,
                     force_door_open=False):
        # randomly move+pick up keys in the environment
        # randomly open/unlock door
        
        split_idx = self.env.unwrapped.splitIdx
        height = self.env.unwrapped.height - 1
        
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
            pos = key.position
            all_pos.remove(pos)
            
            randval = np.random.rand()
            if ((randval < 0.1) or (key.colour == pickup_colour)) and (key.colour != keep_colour):
                key_obj = self.env.unwrapped.grid.get(pos[0], pos[1])
                self.env.unwrapped.carrying.append(key_obj)
                key_obj.cur_pos = np.array([-1, -1])
                self.env.unwrapped.grid.set(pos[0], pos[1], None)
                
            else:
                new_pos_found = False
                while new_pos_found is False:
                    new_pos = (np.random.randint(1, split_idx), np.random.randint(1,height))
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
        self.obs = deepcopy(obs)
        self._timestep += 1
        info = self._modify_info_dict(info)
        
        # if type(obs) is not np.ndarray:
        #     obs = obs.numpy()
        
        # if self.state_size is not None:
        #     img = Image.fromarray(obs)
        #     obs = np.asarray(img.resize(self.state_size, Image.BICUBIC))
        
        # obs = torch.from_numpy(obs)
        # obs = obs.permute((2,0,1)).float()
        
        if self.check_option_complete(self):
            return obs, 1, True, info
        else:
            return obs, 0, done, info

# class LockedRoomPolicyTrainWrapper(Wrapper):
#     def __init__(self, 
#                  env: Env,
#                  check_option_complete=lambda x: False,
#                  option_reward=1.0,
#                  door_colours=[],
#                  key_colour=None,
#                  key_collected=False,
#                  door_unlocked=False,
#                  time_limit=2000):
#         super().__init__(env)
        
#         self.objects = {
#             "doors": [],
#             "key": None,
#             "goal": None
#         }
#         self.door_colours = door_colours
#         self.key_colour = key_colour
#         self.key_door_position = None
#         self._timestep = 0
#         self.time_limit = time_limit
#         self.key = None
        
#         self.check_option_complete = check_option_complete
#         self.option_reward = option_reward
        
#         self.key_collected = key_collected
#         self.door_unlocked = door_unlocked
    
#     def _modify_info_dict(self, info):
#         info['timestep'] = self._timestep
#         info['key'] = self.objects["key"]
#         info['doors'] = self.objects["doors"]
#         info['goal'] = self.objects["goal"]
        
#         return info
    
#     def _get_key_door(self):
#         for door in self.objects["doors"]:
#             if door.colour == self.objects["key"].colour:
#                 return door
    
#     def _find_objs(self):
#         self.objects = {
#             "doors": [],
#             "key": None,
#             "goal": None
#         }
#         for x in range(self.env.unwrapped.width):
#             for y in range(self.env.unwrapped.height):
#                 cell = self.env.unwrapped.grid.get(x, y)
#                 if cell:
#                     if cell.type == "door":
#                         self.objects["doors"].append(
#                             KeyTuple((x, y), cell.color)
#                         )
#                     elif cell.type == "key":
#                         self.key = cell
#                         self.objects["key"] = DoorTuple((x, y), cell.color)
#                     elif cell.type == "goal":
#                         self.objects["goal"] = (x, y)
#                     elif cell.type == "wall":
#                         continue
#                     else:
#                         raise Exception("Unrecognized object {} found at ({},{})".format(cell, x, y))
        
#         if self.key_door_position is None:
#             key_door = self._get_key_door()
#             self.key_door_position = key_door.position
    
#     def _set_key_colour(self):
#         if self.key_colour is None:
#             self.key_colour = self.objects["key"].colour
#             return
        
#         new_key = Key(self.key_colour)
        
#         self.env.unwrapped.grid.set(
#             self.objects["key"].position[0],
#             self.objects["key"].position[1],
#             new_key
#         )
        
#         old_colour = self.objects["key"].colour
#         self.objects["key"] = KeyTuple(self.objects["key"].position,
#                                        self.key_colour)
        
#         new_door = Door(self.key_colour)
#         doors = []
#         for door in self.objects["doors"]:
#             if door.colour == old_colour:
#                 self.env.unwrapped.grid.set(
#                     door.position[0],
#                     door.position[1],
#                     new_door
#                 )
#                 doors.append(DoorTuple(door.position,
#                                        self.key_colour))
#             elif door.colour == self.key_colour:
#                 replace_door = Door(old_colour)
#                 self.env.unwrapped.grid.set(
#                     door.position[0],
#                     door.position[1],
#                     replace_door
#                 )
#                 doors.append(DoorTuple(door.position,
#                                        old_colour))
#             else:
#                 doors.append(door)
#         self.objects["doors"] = doors
    
#     def _set_door_colours(self):
#         if len(self.door_colours) == 0:
#             return
        
#         c_idx = 0
#         for idx, door in enumerate(self.objects["doors"]):
#             if door.colour == self.key_colour:
#                 continue
#             if c_idx >= len(self.door_colours):
#                 return
            
#             colour = self.door_colours[c_idx]
            
#             new_door = Door(colour)
#             self.env.unwrapped.grid.set(
#                 door.position[0],
#                 door.position[1],
#                 new_door
#             )
#             self.objects["doors"][idx] = DoorTuple(door.position,
#                                                    colour)
#             c_idx += 1
            
    
#     def _get_key_door(self):
#         for door in self.objects["doors"]:
#             if door.colour == self.objects["key"].colour:
#                 return door
    
#     def reset(self,
#               agent_reposition_attempts=0,
#               random_start=False,
#               agent_position=None,
#               random_doors_open=[],
#               random_doors_closed=[],
#               random_force_key_pickedup=None,
#               random_force_door_unlocked=None,
#               random_move_agent=False):
#         obs, info = self.env.reset()
        
#         self._find_objs()
#         self._set_key_colour()
#         self._set_door_colours()
        
#         if self.key_collected:
#             key = self.env.unwrapped.grid.get(self.objects["key"].position[0],
#                                               self.objects["key"].position[1])
#             self.env.unwrapped.carrying = key
#             key.cur_pos = np.array([-1,-1])
#             self.env.unwrapped.grid.set(self.objects["key"].position[0],
#                                         self.objects["key"].position[0],
#                                         None)
#         if self.door_unlocked:
#             correct_door = self._get_key_door()
#             door = self.env.unwrapped.grid.get(correct_door.position[0],
#                                                correct_door.position[1])
#             door.is_locked = False
        
#         if agent_position is not None:
#             if self.env.unwrapped.grid.get(agent_position[0],
#                                            agent_position[1]) is None:
#                 self.env.unwrapped.agent_pos = agent_position
        
#         if random_start is True:
#             self.random_start(doors_open=random_doors_open,
#                               doors_closed=random_doors_closed,
#                               force_key_pickedup=random_force_key_pickedup,
#                               force_door_unlocked=random_force_door_unlocked,
#                               randomly_place_agent=random_move_agent)
        
#         self._find_objs()
        
#         obs, _, _, info = self.env.step(actions.LEFT)
#         obs, _, _, info = self.env.step(actions.RIGHT)
        
#         self.env.unwrapped.time_step = 0
#         self._timestep = 0
        
#         info = self._modify_info_dict(info)
        
#         if type(obs) is np.ndarray:
#             obs = torch.from_numpy(obs).float()
        
#         return obs, info
    
#     def random_start(self, 
#                      doors_open=[],
#                      doors_closed=[],
#                      force_key_pickedup=None,   # false key not picked up true key picked up
#                      force_door_unlocked=None,
#                      randomly_place_agent=False):
#         for door in self.objects["doors"]:
#             randval = np.random.rand()
#             if randval < 0.5 and door.colour is not self.key_colour:
#                 door_obj = self.env.unwrapped.grid.get(
#                     door.position[0],
#                     door.position[1]
#                 )
#                 door_obj.is_open = True
            
#             if force_door_unlocked is None or force_door_unlocked is True:
#                 if (door.colour is self.key_colour and randval < 0.3) or force_door_unlocked is True:
#                     door_obj = self.env.unwrapped.grid.get(
#                         door.position[0],
#                         door.position[1]
#                     )
#                     door_obj.is_locked = False
                    
#                     if randval < 0.15:
#                         door_obj.is_open = True
#             if door.colour in doors_open:
#                 door_obj = self.env.unwrapped.grid.get(
#                     door.position[0],
#                     door.position[1]
#                 )
#                 door_obj.is_open = True
#             if door.colour in doors_closed:
#                 door_obj = self.env.unwrapped.grid.get(
#                     door.position[0],
#                     door.position[1]
#                 )
#                 door_obj.is_open = False
        
#         randval = np.random.rand()
#         if (force_key_pickedup is True or randval < 0.3) and (force_key_pickedup is not False):
#             key_obj = self.env.unwrapped.grid.get(self.objects["key"].position[0],
#                                                     self.objects["key"].position[1])
#             key_obj.cur_pos = np.array([-1, -1])
#             self.env.unwrapped.carrying = key_obj
#             self.env.unwrapped.grid.set(self.objects["key"].position[0],
#                                         self.objects["key"].position[1],
#                                         None)
#         else:
#             new_pos_found = False
#             while new_pos_found is False:
#                 new_pos = (np.random.randint(1, 18),
#                             np.random.randint(1, 18))
                
#                 obj = self.env.unwrapped.grid.get(new_pos[0],
#                                                     new_pos[1])
#                 if obj is None:
#                     key_obj = self.env.unwrapped.grid.get(self.objects["key"].position[0],
#                                                             self.objects["key"].position[1])
#                     self.env.unwrapped.grid.set(self.objects["key"].position[0],
#                                                 self.objects["key"].position[1],
#                                                 None)
#                     self.env.unwrapped.grid.set(new_pos[0],
#                                                 new_pos[1],
#                                                 key_obj)
#                     new_pos_found = True
        
#         if randomly_place_agent is True:
#             new_pos_found = False
#             while new_pos_found is False:
#                 new_pos = (np.random.randint(1, 18),
#                            np.random.randint(1, 18))
#                 obj = self.env.unwrapped.grid.get(new_pos[0],
#                                                   new_pos[1])
#                 if obj is None:
#                     self.env.unwrapped.agent_pos = new_pos
#                     self.env.unwrapped.agent_dir = np.random.randint(0,4)
#                     new_pos_found = True
        
    
#     def step(self, action):
#         obs, reward, done, info = self.env.step(action)
#         self._timestep += 1
#         info = self._modify_info_dict(info)
#         if self._timestep >= self.time_limit:
#             done = True
#         if type(obs) is np.ndarray:
#             obs = torch.from_numpy(obs).float()
#         if self.check_option_complete(self):
#             return obs, 1, True, info
#         else:
#             return obs, 0, done, info


# class CorridorWrapper(Wrapper):
#     def __init__(self, env: Env):
#         super().__init__(env)
    
#     def _find_key(self):
#         for x in range(self.env.unwrapped.width):
#             for y in range(self.env.unwrapped.height):
#                 cell = self.env.unwrapped.grid.get(x,y)
#                 if cell:
#                     if cell.type == "key":
#                         return x, y, cell
    
#     def _modify_info_dict(self, info):
#         info['timestep'] = self._timestep
#         info['player_dir'] = self.env.unwrapped.agent_dir
        
#         return info
    
#     def reset(self, random_start=None):
        
#         obs, info = self.env.reset()
        
        
#         x, y, key = self._find_key()
#         key.cur_pos = np.array([-1,-1])
#         self.env.unwrapped.grid.set(x, y, None)
#         self.env.unwrapped.carrying = key
        
#         self.env.unwrapped.agent_pos = (3,5)
        
#         obs, _, _, info = self.env.step(actions.LEFT)
#         obs, _, _, info = self.env.step(actions.RIGHT)
        
#         self.env.unwrapped.time_step = 0
#         self._timestep = 0
        
#         info = self._modify_info_dict(info)
        
#         return obs, info
    
#     def available_actions(self):
#         mask = [torch.tensor(True, dtype=bool)]*7
        
#         agent_dir = self.env.unwrapped.agent_dir
#         if agent_dir == 0 or agent_dir == 2:
#             mask[2] = torch.tensor(False, dtype=bool)
        
#         mask[3] = torch.tensor(False, dtype=bool)
#         mask[4] = torch.tensor(False, dtype=bool)
#         mask[5] = torch.tensor(False, dtype=bool)
#         mask[6] = torch.tensor(False, dtype=bool)
        
#         return mask
    
#     def step(self, action):
#         obs, reward, done, info = self.env.step(action)
        
#         _, agent_y = self.env.unwrapped.agent_pos
        
#         if agent_y == 1:
#             reward = 1
#             done = True
#         else:
#             reward = 0
        
#         info = self._modify_info_dict(info)
        
#         return obs, reward, done, info

