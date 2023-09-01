import os
import lzma
import dill
import itertools
from collections import deque
from contextlib import contextmanager

import torch
import torch.nn as nn
import numpy as np
from pfrl import replay_buffers
from pfrl.replay_buffer import batch_experiences
from pfrl.utils.batch_states import batch_states
from portable.option.policy.models.ppo_mlp import PPOMLP
from experiments.procgen.ppo.ppo_ensemble import PPOEnsemble

class PPOEnsemble():
    def __init__(self,
                 use_gpu,
                 num_actions,
                 phi,
                 
                 attention_module_num=3,
                 learning_rate=5e-4,
                 warmup_size=1024,
                 batch_size=32):
        self.use_gpu = use_gpu
        self.num_actions = num_actions
        
    def _create_agent(self):
        policy = PPOMLP(self.num_actions)
        
    
    def move_to_gpu(self):
        pass
    
    def move_to_cpu(self):
        pass
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass
    
    def step(self):
        pass
    
    def update_accumulated_reward(self, reward):
        pass
    
    def update_leader(self):
        pass
    
    def train(self):
        pass
    
    def predict_actions(self):
        pass
    
    






