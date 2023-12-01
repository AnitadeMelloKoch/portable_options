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
from torch.utils.tensorboard import SummaryWriter
from collections import deque

from portable.option import AttentionOption
from portable.option.ensemble.custom_attention import MockAutoEncoder

from portable.agent.option_agent import OptionAgent

from experiments.experiment_logger import VideoGenerator

@gin.configurable
class AdvancedMinigridFactoredExperiment():
    def __init__(self,
                 base_dir,
                 experiment_name,
                 training_seed,
                 experiment_seed,
                 markov_option_builder,
                 policy_phi,
                 num_instances_per_option=10,
                 num_primitive_actions=7,
                 policy_lr=1e-4,
                 policy_max_steps=1e6,
                 policy_success_threshold=0.98,
                 use_gpu=True,
                 names=None,
                 use_oracle_for_term=True,
                 termination_oracles=None,
                 make_videos=False):

        # for now this is only testing policy when given a factored state 
        # representation as the "embedding"
        
        self.training_seed=training_seed
        self.policy_lr=policy_lr
        self.policy_max_steps=policy_max_steps
        self.policy_success_threshold=policy_success_threshold
        self.use_gpu=use_gpu
        self.num_primitive_actions=num_primitive_actions
        self.num_intances_per_option=num_instances_per_option
        
        # have to use oracle for now
        self.use_oracle_for_term=True
        assert termination_oracles is not None
        self.termination_oracles = termination_oracles
        
        self.names=names
        self.embedding = MockAutoEncoder()
        
        set_seed(experiment_seed)
        self.seed = experiment_seed
        self.name = experiment_name
        self.base_dir = os.path.join(base_dir, experiment_name, str(experiment_seed))
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.plot_dir = os.path.join(self.base_dir, 'plots')
        self.save_dir = os.path.join(self.base_dir, 'checkpoints')
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=self.log_dir)
        log_file = os.path.join(self.log_dir, 
                                "{}.log".format(datetime.datetime.now()))
        
        if make_videos:
            self.video_generator = VideoGenerator(os.path.join(self.base_dir, "videos"))
        else:
            self.video_generator = None
        
        logging.basicConfig(filename=log_file, 
                            format='%(asctime)s %(levelname)s: %(message)s',
                            level=logging.INFO)
        logging.info("[experiment] Beginning experiment {} seed {}".format(self.name, self.seed))
        logging.info("======== HYPERPARAMETERS ========")
        logging.info("Experiment seed: {}".format(experiment_seed))
        logging.info("Training seed: {}".format(training_seed))
        
        self.option = AttentionOption(use_gpu=use_gpu,
                                      log_dir=os.path.join(self.log_dir, 'option'),
                                      markov_option_builder=markov_option_builder,
                                      embedding=self.embedding,
                                      policy_phi=policy_phi,
                                      num_actions=num_primitive_actions,
                                      use_oracle_for_term=self.use_oracle_for_term,
                                      termination_oracle=termination_oracles,
                                      save_dir=self.save_dir,
                                      video_generator=self.video_generator,
                                      option_name=names)
    
    def _video_log(self, line):
        if self.video_generator is not None:
            self.video_generator.add_line(line)
    
    def save(self):
        self.option.save()
    
    def load(self):
        self.option.load()
    
    def train_policy(self,
                     training_envs):
        self.option.bootstrap_policy(training_envs,
                                     self.policy_max_steps,
                                     self.policy_success_threshold)
        
        self.option.save()
    
    def run_episode(self,
                    env):
        
        obs, info = env.reset()
        done = False
        episode_reward = 0
        rewards = []
        steps = 0
        
        if self.video_generator is not None:
            self.video_generator.episode_start()
        
        while not done:
            self.option.run(env, 
                            obs, 
                            info,
                            eval=True,
                            false_states=[])
            
            
            
        
