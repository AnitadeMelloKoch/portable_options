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

def resnet_cnn(num_actions):
    return ResnetCNN(num_actions)

class ResidualBlock(nn.Module):
    def __init__(self, out_channels, use_layer_norm=False):
        super().__init__()
        self.inner_op1 = nn.LazyConv2d(out_channels=out_channels, kernel_size=3, padding=1)
        self.inner_op2 = nn.LazyConv2d(out_channels=out_channels, kernel_size=3, padding=1)
        self.non_linearity = nn.ReLU() 
        self.use_layer_norm = use_layer_norm
        
        if self.use_layer_norm:
            self.norm1 = nn.LazyBatchNorm2d()
            self.norm2 = nn.LazyBatchNorm2d()
    
    def __call__(self, x):
        out = x
        if self.use_layer_norm:
            out = self.norm1(out)
        out = self.non_linearity(out)
        out = self.inner_op1(out)
        
        if self.use_layer_norm:
            out = self.norm2(out)
        out = self.non_linearity(out)
        out = self.inner_op2(out)
        
        return x+out
            
class DownSample(nn.Module):
    def __init__(self, out_channels):
        super().__init__()   
        self.model = nn.Sequential(
            nn.LazyConv2d(
                out_channels=out_channels,
                kernel_size=3,
                stride=1
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
    
    def __call__(self, x):
        return self.model(x)

class ResnetCNN(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions
        
        self.model = nn.Sequential(
            DownSample(16),
            ResidualBlock(16),
            ResidualBlock(16),
            
            DownSample(32),
            ResidualBlock(32),
            ResidualBlock(32),
            
            DownSample(32),
            ResidualBlock(32),
            ResidualBlock(32),
            
            nn.ReLU(),
            
            nn.Flatten(),
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.LazyLinear(num_actions),
            
            pfrl.q_functions.DiscreteActionValueHead()
        )
    
    def __call__(self, x):
        return self.model(x)
    
    

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
                 target_update_interval=100,
                 summary_writer=None):
        # model = create_cnn(num_actions)
        model = resnet_cnn(num_actions)
        
        opt = optim.Adam(model.parameters(), lr=learning_rate)
        explorer = explorers.LinearDecayEpsilonGreedy(start_epsilon,
                                                      end_epsilon,
                                                      epsilon_decay_steps,
                                                      lambda:torch.randint(0, num_actions, size=(1,)))
        
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
        
        self.writer = summary_writer
        
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
        
        if self.writer is not None:
            q_vals = self.agent.model(self.phi(obs).unsqueeze(0).to(self.agent.device).float()).q_values
            self.writer.add_scalar("ddqn/max-q-val", q_vals.max().item(), self.step)
            self.writer.add_scalar("ddqn/mean-q-val", q_vals.mean().item(), self.step)
            if len(self.agent.loss_record) > 0:
                self.writer.add_scalar("ddqn/loss", self.agent.loss_record[-1], self.step)
        
        return out
        
    def observe(self, obs, reward, done, reset):
        if type(obs) == np.ndarray:
            obs = torch.from_numpy(obs)
        obs = obs.to(torch.uint8)
        self.agent.observe(obs,
                           reward,
                           done,
                           reset)




