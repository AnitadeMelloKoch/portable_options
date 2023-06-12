import torch 
import random 
import numpy as np 
import os 
import logging 
import datetime 
import gin 
from portable.option.sets.utils import VOTE_FUNCTION_NAMES

@gin.configurable
class BaseExperiment():
    def __init__(self,
                 base_dir,
                 experiment_name,
                 random_seed,
                 create_env_function,
                 device_type="cpu"):
        
        assert device_type in ["cpu", "cuda"]
        
        self.device = torch.device(device_type)
        
        self.make_env = create_env_function
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        self.random_seed = random_seed
        self.name = experiment_name
        self.base_dir = os.path.join(base_dir, self.name, str(self.random_seed))
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.plot_dir = os.path.join(self.base_dir, 'plots')
        self.save_dir = os.path.join(self.base_dir, 'checkpoints')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        
        
        


