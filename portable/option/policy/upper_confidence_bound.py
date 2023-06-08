import numpy as np
import os
import torch

class UpperConfidenceBound():
    def __init__(self,
                 num_modules,
                 device,
                 c=100):
        
        self.num_modules = num_modules
        self.device = device
        self.c = c

        # value in bandit problem
        self.accumulated_reward = np.ones(num_modules)
        self.time_step = 0
        self.visitation_count = np.zeros(num_modules)

    @staticmethod
    def _get_save_dirs(path):
        return os.path.join(path, 'accumulated_reward.npy'), \
            os.path.join(path, 'time_step.npy'), \
            os.path.join(path, 'visitation_count.npy')

    def save(self, path):
        os.makedirs(path, exist_ok=True)

        reward_file, time_step_file, count_file = self._get_save_dirs(path)

        np.save(reward_file, self.accumulated_reward)
        np.save(time_step_file, self.time_step)
        np.save(count_file, self.visitation_count)

    def load(self, path):
        reward_file, time_step_file, count_file = self._get_save_dirs(path)
        if not os.path.exists(reward_file):
            return
        if not os.path.exists(time_step_file):
            return
        if not os.path.exists(count_file):
            return
        
        self.accumulated_reward = np.load(reward_file)
        self.time_step = np.load(time_step_file)
        self.visitation_count = np.load(count_file)

    def weights(self, move_to_device=True):
        """
        Return the weights of all members of the ensemble at this time step.
        """
        weights = self.accumulated_reward + self.c*np.sqrt(
            2*np.log(self.time_step)/(self.visitation_count+1e-8)
        )
        if move_to_device:
            return torch.from_numpy(weights).to(self.device)
        else:
            return weights

    def select_leader(self):
        """
        Select leader using upper confidence bound. Incriments visitation count.
        """
        leader = np.argmax(self.weights(False))
        self.visitation_count[leader] += 1

        return leader

    def step(self):
        self.time_step += 1

    def update_accumulated_rewards(self, learner_idx, reward):
        self.accumulated_reward[learner_idx] += reward

