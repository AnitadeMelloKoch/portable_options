import logging 
import datetime 
import os 
import random 
import gin 
import torch 
import lzma 
import dill 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from portable.utils.utils import set_seed

from portable.option.sets.utils import VOTE_FUNCTION_NAMES

@gin.configurable
class AdvancedMinigridExperiment():
    def __init__(self,
                 base_dir,
                 experiment_name,
                 training_seed,
                 experiment_seed,
                 create_agent_function,
                 action_space,
                 num_options,
                 lr=1e-4,
                 initiation_vote_function="weighted_vote_low",
                 termination_vote_function="weighted_vote_low",
                 device_type="cpu",
                 sigma=0.5):
        
        assert device_type in ["cpu", "cuda"]
        assert initiation_vote_function in VOTE_FUNCTION_NAMES
        assert termination_vote_function in VOTE_FUNCTION_NAMES
        
        self.device = torch.device(device_type)
        self.training_seed = training_seed
        
        self.agent = create_agent_function(action_space,
                                           gpu=0,
                                           n_input_channels=3,
                                           lr=lr,
                                           sigma=sigma)
        
        set_seed(experiment_seed)
        self.seed = experiment_seed
        self.name = experiment_name
        self.base_dir = os.path.join(base_dir, experiment_name, experiment_seed)
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.plot_dir = os.path.join(self.base_dir, 'plots')
        self.save_dir = os.path.join(self.base_dir, 'checkpoints')
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        
        log_file = os.path.join(self.log_dir, "{}.log".format(datetime.datetime.now()))
        logging.basicConfig(filename=log_file, 
                            format='%(asctime)s %(levelname)s: %(message)s')
        logging.info("[experiment] Beginning experiment {} seed {}".format(self.name, self.random_seed))
        logging.info("======== HYPERPARAMETERS ========")
        logging.info("Training base rainbow agent with no options")
        logging.info("Experiment seed: {}".format(experiment_seed))
        logging.info("Training seed: {}".format(training_seed))
        logging.info("Test seeds: {}".format(self.test_env_seeds))
        
        self.trial_data = pd.DataFrame([],
                                       columns=['reward',
                                                'seed',
                                                'frames',
                                                'env_num'])
        # add options!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.options = []
        
    def save(self):
        self.agent.save()
        os.makedirs(self.save_dir, exist_ok=True)
        filename = os.path.join(self.save_dir, 'experiment_data.pkl')
        with lzma.open(filename, 'wb') as f:
            dill.dump(self.trial_data, f)

    def load(self):
        self.agent.load()
        filename = os.path.join(self.save_dir, 'experiment_data.pkl')
        if os.path.exists(filename):
            with lzma.open(filename, 'rb') as f:
                self.trial_data = dill.load(f)

    def train_options(self):
        # train all options
        pass
    
    def run_episode(self):
        # run a single experiment
        pass
    
    def run(self,
            make_env,
            num_envs,
            frames_per_env):
        # run experiment
        pass