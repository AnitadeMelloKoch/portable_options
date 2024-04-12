from enum import IntEnum

import numpy as np
import gym
from gym import spaces
from skimage import color


class actions(IntEnum):
    INVALID         = -1
    NOOP            = 0
    FIRE            = 1
    UP              = 2
    RIGHT           = 3
    LEFT            = 4
    DOWN            = 5
    UP_RIGHT        = 6
    UP_LEFT         = 7
    DOWN_RIGHT      = 8
    DOWN_LEFT       = 9
    UP_FIRE         = 10
    RIGHT_FIRE      = 11
    LEFT_FIRE       = 12
    DOWN_FIRE       = 13
    UP_RIGHT_FIRE   = 14
    UP_LEFT_FIRE    = 15
    DOWN_RIGHT_FIRE = 16
    DOWN_LEFT_FIRE  = 17
    

class MontezumaPortWrapper(gym.Wrapper):
    def __init__(self, env, max_steps=60*60*30):
        self.T = 0
        self.num_lives = None
        self.lost_life = False
        self.added_rewards = {}
        self.episode_rewards = self.added_rewards.copy()
        self._elapsed_steps = 0
        self._max_episode_steps = max_steps
        gym.Wrapper.__init__(self, env)

    def reset(self, **kwargs):
        s0 = self.env.reset(**kwargs)
        self.num_lives = self.get_num_lives(self.get_current_ram())
        info = self.get_current_info(info={})
        self.episode_rewards = self.added_rewards.copy()
        self._elapsed_steps = 0
        
        return s0, info

    def step(self, action):
        self.T += 1
        self.lost_life = False
        obs, reward, done, info = self.env.step(action)
        reward += self.check_reward()
        info = self.get_current_info(info=info)
        if self.num_lives is not None and self.num_lives > info["lives"]:
            self.episode_rewards = self.added_rewards.copy()
            self.lost_life = True

        self.num_lives = info["lives"]

        self._elapsed_steps += 1

        if self._max_episode_steps <= self._elapsed_steps:
            info["needs_reset"] = True

        return obs, reward, done, info

    def add_reward(self, x, y, room, reward):
        self.added_rewards[(x,y,room)] = reward

    def check_reward(self):
        ram = self.get_current_ram()

        x = self.get_player_x(ram)
        y = self.get_player_y(ram)
        room = self.get_screen_num(ram)

        if (x, y, room) in self.added_rewards:
            reward = self.episode_rewards[(x,y,room)]
            self.episode_rewards[(x,y,room)] = 0
            return reward

        return 0

    def get_is_life_lost(self):
        return self.lost_life

    def get_current_info(self, info):
        ram = self.get_current_ram()
    
        info["lives"] = self.get_num_lives(ram)
        info["falling"] = self.get_is_falling(ram)
        info["player_x"] = self.get_player_x(ram)
        info["player_y"] = self.get_player_y(ram)
        info["dead"] = int(info["lives"] < self.num_lives)
        info["screen_num"] = self.get_screen_num(ram)
        info["jumping"] = self.get_is_jumping(ram)
        info["needs_reset"] = False
        info["elapsed_steps"] = self._elapsed_steps

        return info

    def get_current_position(self):
        ram = self.get_current_ram()
        return self.get_player_x(ram), self.get_player_y(ram), self.get_screen_num(ram)

    def get_player_x(self, ram):
        return int(self.getByte(ram, 'aa'))

    def get_player_y(self, ram):
        return int(self.getByte(ram, 'ab'))

    def get_screen_num(self, ram):
        return int(self.getByte(ram, '83'))

    def get_is_jumping(self, ram):
        return int(int(self.getByte(ram, 'd8')) != 0)

    def get_num_lives(self, ram):
        return int(self.getByte(ram, 'ba'))
    
    def get_is_falling(self, ram):
        return int(int(self.getByte(ram, 'd8')) != 0)

    def get_current_ale(self):
        return self.env.unwrapped.ale
        # return self.env.environment.ale

    def get_current_ram(self):
        return self.get_current_ale().getRAM()

    @staticmethod
    def _getIndex(address):
        assert type(address) == str and len(address) == 2 
        row, col = tuple(address)
        row = int(row, 16) - 8
        col = int(col, 16)
        return row*16+col

    @staticmethod
    def getByte(ram, address):
        # Return the byte at the specified emulator RAM location
        idx = MontezumaPortWrapper._getIndex(address)
        return ram[idx]

    def _get_frame(self):
        img_rgb = np.empty([210, 160, 3], dtype=np.uint8)
        self.env.env.ale.getScreenRGB(img_rgb)

        return img_rgb

    def get_pixels_around_player(self, width=22, height=24, trim_direction=actions.INVALID):
        """
        Extract a window of size (width, height) around the player.
        Args:
            width (int)
            height (int)

        Returns:
            image_window (np.ndarry)
        """
        if trim_direction != actions.INVALID:
            width -= 6
        image = self._get_frame()
        value_to_index = lambda y: int(-1.01144971 * y + 309.86119429)
        player_position = self.get_current_position()
        start_y, end_y = (value_to_index(player_position[1]) - height,
                          value_to_index(player_position[1]) + height)
        start_x, end_x = max(0, player_position[0]+3 - width), player_position[0]+3 + width
        start_y += 0
        end_y += 8
        if trim_direction == actions.RIGHT:
            start_x += 13
            end_x += 13
        elif trim_direction == actions.LEFT:
            start_x -= 7
            end_x -= 7
        image_window = image[start_y:end_y, start_x:end_x, :]

        if ((player_position[0]+3) - width) < 0:
            image_window = np.pad(image_window, ((0,0), (abs((player_position[0]+3) - width), 0), (0,0)))

        if image_window.shape[1] != (2*width):
            image_window = np.pad(image_window, ((0,0), (0, (2*width) - image_window.shape[1]), (0,0)))
        
        return image_window

class ReshapeFrame(gym.ObservationWrapper):
    def __init__(self, env, channel_order="hwc"):
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        shape = {
            "hwc": (self.height, self.width, 1),
            "chw": (1, self.height, self.width),
        }
        self.observation_space = spaces.Box(
            low=0, high=255, shape=shape[channel_order], dtype=np.uint8
        )

    def observation(self, frame):
        return frame.reshape(self.observation_space.low.shape)