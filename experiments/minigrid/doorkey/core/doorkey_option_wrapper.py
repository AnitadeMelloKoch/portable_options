from gymnasium.core import Wrapper
from experiments.minigrid.utils import actions
from enum import IntEnum
import matplotlib.pyplot as plt

class directions(IntEnum):
    RIGHT       = 0
    DOWN          = 1
    LEFT        = 2
    UP        = 3

# 7 => go to key
### can execute if player has not yet collected the key
# 8 => go to door
### can execute always
# 9 => go to goal
### can execute if door open

class DoorKeyEnvOptionWrapper(Wrapper):
    """Add perfect options that the agent has access to through normal action api"""
    
    def __init__(self, env):
        super().__init__(env)
        
        self.states = []
        self.rewards = []
        self.infos = []
        self.door_pos = ()
        self.key_pos = ()
        self.goal_pos = ()
        
        self._find_objs()
    
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
    
    def available_actions(self):
        available_actions = [True]*10
        if self.env.unwrapped.carrying:
            available_actions[7] = False
        if not self.env.unwrapped.grid.get(self.door_pos[0], self.door_pos[1]).is_open():
            available_actions[9] = False
        
        return available_actions
    
    def _move(self, x, y):
        player_x = self.env.agent_pos[0]
        player_y = self.env.agent_pos[1]
        
        """
        Travel to correct x position
        """
        if player_x > x:
            while self.env.unwrapped.agent_dir != directions.LEFT:
                obs, reward, done, info = self.env.step(actions.LEFT)
                self.states.append(obs)
                self.rewards.append(reward)
                self.infos.append(info)
                if done:
                    return obs, reward, done, info
        elif player_x < x:
            while self.env.unwrapped.agent_dir != directions.RIGHT:
                obs, reward, done, info = self.env.step(actions.RIGHT)
                self.states.append(obs)
                self.rewards.append(reward)
                self.infos.append(info)
                if done:
                    return obs, reward, done, info
        
        for _ in range(abs(player_x-x)):
            obs, reward, done, info = self.env.step(actions.FORWARD)
            self.states.append(obs)
            self.rewards.append(reward)
            self.infos.append(info)
            if done:
                return obs, reward, done, info
        
        """
        Travel to correct y position
        """
        if player_y > y:
            while self.env.unwrapped.agent_dir != directions.UP:
                obs, reward, done, info = self.env.step(actions.LEFT)
                self.states.append(obs)
                self.rewards.append(reward)
                self.infos.append(info)
                if done:
                    return obs, reward, done, info
        elif player_y < y:
            while self.env.unwrapped.agent_dir != directions.DOWN:
                obs, reward, done, info = self.env.step(actions.RIGHT)
                self.states.append(obs)
                self.rewards.append(reward)
                self.infos.append(info)
                if done:
                    return obs, reward, done, info
        
        for _ in range(abs(player_y-y)):
            obs, reward, done, info = self.env.step(actions.FORWARD)
            self.states.append(obs)
            self.rewards.append(reward)
            self.infos.append(info)
            if done:
                return obs, reward, done, info
        
        obs, reward, done, info = self.env.step(actions.PICKUP)
        
        return obs, reward, done, info
    
    def _go_key(self):
        if self.env.unwrapped.carrying:
            ## If option not available execute a NO-OP
            return self.env.step(actions.PICKUP)
            
        obs, reward, done, info = self._move(self.key_pos[0], self.key_pos[1])
        
        return obs, reward, done, info
    
    def _go_door(self):
        obs, reward, done, info = self._move(self.door_pos[0], self.door_pos[1]) 
        
        return obs, reward, done, info
    
    def _go_goal(self):
        if not self.env.unwrapped.grid.get(self.door_pos[0], self.door_pos[1]).is_open:
            # if option not available execute a NO-OP
            return self.env.step(actions.PICKUP)
        player_x = self.env.agent_pos[0]
        player_y = self.env.agent_pos[1]
        if player_x < self.door_pos[0]:
            obs, reward, done, info = self._move(self.door_pos[0], self.door_pos[1])
            
        obs, reward, done, info = self._move(self.goal_pos[0], self.goal_pos[1])
        
        return obs, reward, done, info
    
    def step(self, action):
        assert action < 10
        
        if action < 7:
            obs, reward, done, info = self.env.step(action)
        elif action == 7:
            obs, reward, done, info = self._go_key()
        elif action == 8:
            obs, reward, done, info = self._go_door()
        elif action == 9:
            obs, reward, done, info = self._go_goal()
        else:
            raise
        
        info["option_states"] = self.states
        self.states = []
        info["option_rewards"] = self.rewards
        if len(self.rewards) > 0:
            reward = sum(self.rewards)
        self.rewards = []
        info["option_infos"] = self.infos
        self.infos = []
        
        return obs, reward, done, info
    
    