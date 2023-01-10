from agents.bonus_based_exploration import RNDRainbowAgentForOptions

from agents.bonus_based_exploration.run_experiment import create_exploration_agent, create_exploration_runner
from dopamine.discrete_domains import run_experiment
import numpy as np
import os
import logging

## do logging

from portable.option.sets.utils import get_vote_function, VOTE_FUNCTION_NAMES
import torch
import random

class RainbowAgent():

    def __init__(self,
                 base_dir,
                 seed,
                 experiment_name,
                 action_num,
                 initiation_vote_function,
                 termination_vote_function,
                 policy_phi,
                 experiment_env_function,
                 device_type="cpu",
                 train_initiation=True,
                 options_initiation_positive_files=[[]],
                 options_initiation_negative_files=[[]],
                 options_initiation_priority_negative_files=[[]],
                 train_initiation_embedding_epochs=50,
                 train_initiation_classifier_epochs=50,
                 train_termination=True,
                 options_termination_positive_files=[[]],
                 options_termination_negative_files=[[]],
                 options_termination_priority_negative_files=[[]],
                 train_termination_embedding_epochs=50,
                 train_termination_classifier_epochs=50,
                 train_policy=True,
                 policy_bootstrap_env=None,
                 train_policy_max_steps=10000,
                 train_policy_success_rate=0.9,
                 ):
        assert device_type in ["cpu", "cuda"]
        assert initiation_vote_function in VOTE_FUNCTION_NAMES
        assert termination_vote_function in VOTE_FUNCTION_NAMES

        self.device = torch.device(device_type)
        
        random.seed(seed)
        self.seed = seed
        self.name = experiment_name
        self.base_dir = os.path.join(base_dir, self.name, str(self.seed))
        self.log_dir = os.path.join(self.base_dir, "logs")
        self.plot_dir = os.path.join(self.base_dir, "plots")
        self.save_dir = os.path.join(self.base_dir, "checkpoints")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)



