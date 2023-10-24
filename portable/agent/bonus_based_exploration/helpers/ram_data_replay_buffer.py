import os
from absl import logging
import numpy as np

class MontezumaRevengeReplayBuffer():
    def __init__(self, replay_capacity=1000000):

        self.replay_capacity = replay_capacity
        self._create_storage()
        self.cursor_count = 0

    def _create_storage(self):
        self.memory = {}

        self.memory['player_x'] = np.empty([self.replay_capacity], dtype=np.int)
        self.memory['player_y'] = np.empty([self.replay_capacity], dtype=np.int)
        self.memory['room_number'] = np.empty([self.replay_capacity], dtype=np.int)
        obs_array_shape = [self.replay_capacity] + list((84,84))
        self.memory['observation'] = np.empty(obs_array_shape, np.uint8)

    def cursor(self):
        return self.cursor_count % self.replay_capacity

    def add(self, player_x, player_y, room_number, observation):

        self.memory['player_x'][self.cursor()] = player_x
        self.memory['player_y'][self.cursor()] = player_y
        self.memory['room_number'][self.cursor()] = room_number
        self.memory['observation'][self.cursor()] = observation

        self.cursor_count += 1


    def _generate_filename(self, checkpoint_dir, name):
        info_dir = os.path.join(checkpoint_dir, 'ram_info')
        if not os.path.isdir(info_dir):
            os.makedirs(info_dir)
        return os.path.join(info_dir, '{}_ckpt.npy'.format(name))

    def get_index(self, memory_type, index):
        return self.memory[memory_type][index]

    def get_last_index(self):
        if self.cursor_count < self.replay_capacity:
            return self.cursor_count
        else:
            return self.replay_capacity

    def save(self, checkpoint_dir):
        if not os.path.isdir(checkpoint_dir):
            return

        filename = self._generate_filename(checkpoint_dir, 'player_x')
        np.save(filename, self.memory['player_x'], allow_pickle=False)

        filename = self._generate_filename(checkpoint_dir, 'player_y')
        np.save(filename, self.memory['player_y'], allow_pickle=False)

        filename = self._generate_filename(checkpoint_dir, 'room_number')
        np.save(filename, self.memory['room_number'], allow_pickle=False)

        filename = self._generate_filename(checkpoint_dir, 'observation')
        np.save(filename, self.memory['observation'], allow_pickle=False)

    def load(self, checkpoint_dir):

        filename = self._generate_filename(checkpoint_dir, 'player_x')
        if os.path.exists(filename):
            logger.info('Loading player_x data')
            self.memory['player_x'] = np.load(filename, allow_pickle=False)

        filename = self._generate_filename(checkpoint_dir, 'player_y')
        if os.path.exists(filename):
            logger.info('Loading player_y data')
            self.memory['player_y'] = np.load(filename, allow_pickle=False)

        filename = self._generate_filename(checkpoint_dir, 'room_number')
        if os.path.exists(filename):
            logger.info('Loading room_number data')
            self.memory['room_number'] = np.load(filename, allow_pickle=False)

        filename = self._generate_filename(checkpoint_dir, 'observation')
        if os.path.exists(filename):
            logger.info('Loading observation data')
            self.memory['observation'] = np.load(filename, allow_pickle=False)
