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

from portable.option.sets.utils import VOTE_FUNCTION_NAMES

@gin.configurable 
class MinigridExperiment():
    def __init__(self,
                 base_dir,
                 experiment_name,
                 training_seed,
                 random_seed,
                 create_env_function,
                 num_levels,
                 create_agent_function,
                 preprocess_obs,
                 num_frames_train=10**7,
                 initiation_vote_function="weighted_vote_low",
                 termination_vote_function="weighted_vote_low",
                 device_type="cpu",
                 train_options=True,
                 initiation_positive_files=[],
                 initiation_negative_files=[],
                 initiation_priority_negative_files=[],
                 train_initiation_embedding_epochs=100,
                 train_initiation_classifier_epochs=100,
                 termination_positive_files=[],
                 termination_negative_files=[],
                 termination_priority_negative_files=[],
                 train_termination_embedding_epochs=100,
                 train_termination_classifier_epochs=100,
                 policy_bootstrap_envs=[],
                 train_policy_max_steps=10000,
                 train_poilcy_success_rate=0.8,
                 max_option_tries=5):
        
        assert device_type in ["cpu", "cuda"]
        assert initiation_vote_function in VOTE_FUNCTION_NAMES
        assert termination_vote_function in VOTE_FUNCTION_NAMES
        
        self.device = torch.device(device_type)
        
        if train_options:
            self.agent.add_initiation_files(initiation_positive_files,
                                            initiation_negative_files,
                                            initiation_priority_negative_files)
            self.agent.add_termination_files(termination_positive_files,
                                            termination_negative_files,
                                            termination_priority_negative_files)
            
            self.agent.train_options(train_initiation_embedding_epochs,
                                     train_initiation_classifier_epochs,
                                     train_termination_embedding_epochs,
                                     train_termination_classifier_epochs,
                                     policy_bootstrap_envs,
                                     train_policy_max_steps,
                                     train_poilcy_success_rate)    
        
        
        self.make_env = create_env_function
        
        random.seed(random_seed)
        self.random_seed = random_seed
        self.name = experiment_name
        self.base_dir = os.path.join(base_dir, self.name, str(self.random_seed))
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.plot_dir = os.path.join(self.base_dir, 'plots')
        self.save_dir = os.path.join(self.base_dir, 'checkpoints')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        
        training_env = self.make_env(training_seed)
        
        self.agent = create_agent_function(save_dir=os.path.join(self.save_dir, "option_agent"),
                                           device=self.device,
                                           training_env=training_env,
                                           preprocess_obs=preprocess_obs)
        
        self.training_env_seed = training_seed
        self.test_env_seeds = random.choices(np.arange(1000), k=num_levels)
        self.test_env_seeds = [int(x) for x in self.test_env_seeds]
        self.num_levels = num_levels
        self.num_frames_train = num_frames_train
        
        log_file = os.path.join(self.log_dir, "{}.log".format(datetime.datetime.now()))
        logging.basicConfig(filename=log_file, 
                            format='%(asctime)s %(levelname)s: %(message)s')
        logging.info("[experiment] Beginning experiment {} seed {}".format(self.name, self.random_seed))
        logging.info("======== HYPERPARAMETERS ========")
        logging.info("Train options: {}".format(train_options))
        logging.info("Initiation vote function: {}".format(initiation_vote_function))
        logging.info("Termination vote function: {}".format(termination_vote_function))
        logging.info("Initiation embedding epochs: {}".format(train_initiation_embedding_epochs))
        logging.info("Initiation classifier epochs: {}".format(train_initiation_classifier_epochs))
        logging.info("Termination embedding epochs: {}".format(train_termination_embedding_epochs))
        logging.info("Termination classifier epochs: {}".format(train_termination_classifier_epochs))
        logging.info("Train policy max steps: {}".format(train_policy_max_steps))
        logging.info("Train policy success rate: {}".format(train_poilcy_success_rate))
        logging.info("Max option tries: {}".format(max_option_tries))
        logging.info("Train seed: {}".format(training_seed))
        logging.info("Number of levels in experiment: {}".format(num_levels))
        logging.info("Test seeds: {}".format(self.test_env_seeds))
        
        self.trial_data = pd.DataFrame([],
                                       columns=[
                                           'steps',
                                           'reward'
                                       ],
                                       index=pd.Index([], name='seed'))
        
    def save(self):
        self.agent.save()
        os.makedirs(self.save_dir, exist_ok=True)
        filename = os.path.join(self.save_dir, 'experiment_data.pkl')
        with lzma.open(filename, 'wb') as f:
            dill.dump(self.trial_data, f)
            
    def load(self):
        self.option.load(self.save_dir)
        filename = os.path.join(self.save_dir, 'experiment_data.pkl')
        if os.path.exists(filename):
            with lzma.open(filename, 'rb') as f:
                self.trial_data = dill.load(f)
    
    # def train(self):
    #     for seed in self.test_env_seeds:
    #         env = self.make_env(seed)
    #         logging.info("[train] Starting train with env seed", seed)
    #         print("[train] Starting train with env seed", seed)
            
    #         self.agent.train([env], self.num_frames_train)
    
    def train(self):
        envs = [self.make_env(seed) for seed in self.test_env_seeds]
        logging.info("[train] Starting train ")
        print("[train] Starting train ")
        
        for x in range(5):
            print("Epoch", x)
            self.agent.train(envs, self.num_frames_train)
    
    
    
    
    
    