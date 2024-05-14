import logging 
import os 
import numpy as np 
import gin 
import random 
import torch 
import pickle

from portable.option.divdis.policy.policy_and_initiation import PolicyWithInitiation
from portable.option.policy.agents import evaluating
import matplotlib.pyplot as plt 
from portable.option.policy.intrinsic_motivation.tabular_count import TabularCount

@gin.configurable
class GlobalOption():
    def __init__(self,
                 use_gpu,
                 log_dir,
                 save_dir,
                 policy_phi,
                 video_generator=None):
        self.use_gpu = use_gpu
        self.save_dir = save_dir
        self.policy_phi = policy_phi
        self.log_dir = log_dir
        
        self.policy = PolicyWithInitiation(use_gpu=use_gpu,
                                           policy_phi=policy_phi,
                                           learn_initiation=False)
        
        self.video_generator = video_generator
    
    def _video_log(self, line):
        if self.video_generator is not None:
            self.video_generator.add_line(line)

    def save(self):
        self.policy.save(self.save_dir)
    
    def load(self):
        self.policy.load(self.save_dir)
    
    def train_policy(self,
                     env,
                     obs,
                     info,
                     make_video=False):
        self.policy.move_to_gpu()
        action = self.policy.act(obs)
        
        next_obs, reward, done, info = env.step(action)
        
        self.policy.observe(obs,
                            action,
                            reward,
                            next_obs,
                            done)
        self.policy.move_to_cpu()
        return next_obs, reward, done, info, 1
    
    def eval_policy(self,
                    env,
                    obs,
                    info):
        
        with evaluating(self.policy):
            action = self.policy.act(obs)
            
            next_obs, reward, done, info = env.step(action)
            
            self.policy.observe(obs,
                                action,
                                reward,
                                next_obs,
                                done)

            return next_obs, reward, done, info, 1
    
    
    