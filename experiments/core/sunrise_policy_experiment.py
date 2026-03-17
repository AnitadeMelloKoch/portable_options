import logging  
import datetime
import os 
import gin 
import numpy as np 
from portable.utils.utils import set_seed 
from torch.utils.tensorboard import SummaryWriter
import torch 
import random 
import matplotlib.pyplot as plt 
import pickle 

from portable.option.vf_transfer.vf_transfer_option import VFTransferAgent
import math 

from collections import deque

@gin.configurable
class ValueTransferExperiment():
    def __init__(self,
                 base_dir,
                 experiment_name,
                 seed,
                 policy_phi,
                 gpu_id,
                 discount_rate=0.9):
        
        self.name = experiment_name
        self.seed = seed 
        self.gpu_id = gpu_id
        
        set_seed(seed)
        
        self.base_dir = os.path.join(base_dir, experiment_name, str(seed))
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.save_dir = os.path.join(self.base_dir, 'checkpoints')
        self.plot_dir = os.path.join(self.base_dir, 'plots')
        
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        
        log_file = os.path.join(self.log_dir, "{}.log".format(datetime.datetime.now()))
        logging.basicConfig(filename=log_file,
                            format='%(asctime)s %(levelname)s: %(message)s',
                            level=logging.INFO)
        logging.info(f"[experiment] Beginning experiment {self.name} seed {self.seed}")
        
        self.gamma = discount_rate
        
        self.episode_data = []
        
        self.agent = VFTransferAgent(
            policy_phi=policy_phi,
            log_dir=self.log_dir,
            save_dir=self.save_dir,
            use_gpu=self.gpu_id,
            plot_dir=self.plot_dir
        )
    
    def save(self):
        with open(os.path.join(self.save_dir, 'experiment_results.pkl'), 'wb') as f:
            pickle.dump(self.experiment_data, f)
        
        with open(os.path.join(self.save_dir, 'episode_results.pkl'), 'wb') as f:
            pickle.dump(self.episode_data, f)
        
        self.agent.save(self.save_dir)
    
    def load(self):
        with open(os.path.join(self.save_dir, 'experiment_results.pkl'), 'rb') as f:
            self.experiment_data = pickle.load(f)
        
        with open(os.path.join(self.save_dir, 'episode_results.pkl'), 'rb') as f:
            self.episode_data = pickle.load(f)
        
        self.agent.load(self.save_dir)
        
    def train_agent(self, 
                    env,
                    seed,
                    max_steps):
        total_steps = 0
        episode_rewards = deque(maxlen=200)
        episode = 0
        
        while total_steps < max_steps:                        
            rewards, steps = self.agent.run_episode(seed=seed, env=env)
            total_steps += steps
            sum_reward = np.sum(rewards)
            
            self.episode_data.append({
                "frames": total_steps,
                "episode_length": steps,
                "episode_rewards": rewards,
            })
            
            episode_rewards.append(sum_reward)
            self.writer.add_scalar('episode_rewards', sum_reward, total_steps)
            
            logging.info(f"Episode {episode} total steps: {total_steps} ave undiscounted reward: {np.mean(episode_rewards)}")
            
            episode += 1
            
            
                
                
        
        