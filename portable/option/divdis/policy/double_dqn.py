import os 
import logging 
import gin 
import torch 
import pickle 
import numpy as np
import torch.optim as optim 
from copy import deepcopy
from pfrl import explorers 
from pfrl import replay_buffers 
import pfrl
import torch.nn as nn
from collections import deque 
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt

def create_cnn(num_actions):
    return nn.Sequential(
        nn.LazyConv2d(out_channels=32, kernel_size=12, stride=12),
        nn.ReLU(),
        nn.LazyConv2d(out_channels=64, kernel_size=2, stride=1),
        nn.ReLU(),
        nn.LazyConv2d(out_channels=64, kernel_size=2, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.LazyLinear(512),
        nn.ReLU(),
        nn.LazyLinear(512),
        nn.ReLU(),
        nn.LazyLinear(num_actions),
        
        pfrl.q_functions.DiscreteActionValueHead()
    )

@gin.configurable
class DoubleDQN():
    def __init__(self,
                 use_gpu,
                 learning_rate,
                 phi,
                 num_actions,
                 start_epsilon=1.0,
                 end_epsilon=0.01,
                 epsilon_decay_steps=1e6,
                 buffer_capacity=1e6,
                 minibatch_size=32,
                 replay_start_size=1000,
                 update_interval=1,
                 gamma=0.99,
                 target_update_interval=100):
        model = create_cnn(num_actions)
        
        opt = optim.Adam(model.parameters(), lr=learning_rate)
        explorer = explorers.LinearDecayEpsilonGreedy(start_epsilon,
                                                      end_epsilon,
                                                      epsilon_decay_steps,
                                                      lambda:torch.randint(0, num_actions, size=(1,))
        
        self.agent = pfrl.agents.DoubleDQN(q_function=model,
                                           optimizer=opt,
                                           gpu=use_gpu,
                                           replay_buffer=replay_buffers.PrioritizedReplayBuffer(capacity=buffer_capacity),
                                           gamma=gamma,
                                           explorer=explorer,
                                           minibatch_size=minibatch_size,
                                           replay_start_size=replay_start_size,
                                           update_interval=update_interval,
                                           target_update_interval=target_update_interval,
                                           phi=phi)
        
        self.num_actions = num_actions
        self.step = 0
        self.train_rewards = deque(maxlen=200)
        self.option_runs = 0
        self.phi = phi
        
        logger.info("==================== Policy HPs ====================")
        logger.info(f"learning rate: {learning_rate}")
        logger.info(f"replay buffer capacity: {buffer_capacity}")
        logger.info(f"start epsilon: {start_epsilon}")
        logger.info(f"end epsilon: {end_epsilon}")
        logger.info(f"epsilon decay steps: {epsilon_decay_steps}")
        logger.info(f"minibatch size: {minibatch_size}")
        logger.info(f"update interval: {update_interval}")
        logger.info(f"gamma: {gamma}")
        logger.info(f"target update interval: {target_update_interval}")
        logger.info("====================================================")
    
    def save(self, dir):
        os.makedirs(dir, exist_ok=True)
        self.agent.save(dir)
    
    def load(self, dir):
        print("\033[92m {}\033[00m" .format("DDQN model loaded"))
        self.agent.load(dir)
    
    def act(self, obs):
        self.step += 1
        out = self.agent.act(obs)   
        
        return out
        
    def observe(self, obs, reward, done, reset):
        self.agent.observe(obs,
                           reward,
                           done,
                           reset)




