import logging 
import os 
import numpy as np 
import gin
import random 
import torch 
import pickle 

from collections import deque 

from portable.option.vf_transfer.policy.sunrise import SunriseDQNAgent
from portable.option.policy.agents import evaluating
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

@gin.configurable
class VFTransferAgent():
    def __init__(self,
                 use_gpu,
                 log_dir,
                 save_dir,
                 policy_phi,
                 plot_dir,
                 ):
        
        self.gpu = use_gpu
        self.save_dir = save_dir
        self.log_dir = log_dir,
        self.policy_phi = policy_phi
        self.plot_dir = plot_dir
        
        self.writer = SummaryWriter(log_dir=log_dir)
        
        self.policies = {}
        
        self.seed = None
        self.policy = None
        
        self.train_data = []
        
        self.steps = 0
    
    def _add_policy(self, seed):
        if seed in self.policies:
            return
        self.policies[seed] = SunriseDQNAgent(
            use_gpu=self.gpu,
            policy_phi=self.policy_phi
        )
    
    def save(self):
        pass
    
    def load(self):
        pass
    
    def set_policy_by_seed(self, seed):
        self._add_policy(seed)
        self.seed = seed
        # this should be a reference to the policy
        self.policy = self.policies[seed]
    
    # need to deal with the reward stuff still. This just does a normal training
    # at the moment
    def run_episode(self, seed, env):
        
        steps = 0
        rewards = []
        
        done = False
        
        assert self.seed == seed, "Seeds don't much. Run set_policy_by_seed() first"
        
        state, info = env.reset()
        
        while not done:
            state = self.policy_phi(state)
            action = self.policy.act(state)
            next_state, reward, done, info = env.step(action)
            rewards.append(reward)
            steps += 1
            self.steps += 1
            self.policy.observe(state,
                                action,
                                reward,
                                next_state,
                                done)
            
            state = next_state
        
        if self.writer is not None:
            self.writer.add_scalar('episode_length', steps, self.steps)
            self.writer.add_scalar('episode_rewards', sum(rewards), self.steps)
        
        return rewards, steps
    
                
        
        


