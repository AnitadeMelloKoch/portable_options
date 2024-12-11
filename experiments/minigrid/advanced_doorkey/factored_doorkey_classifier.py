from experiments.minigrid.advanced_doorkey.core.advanced_minigrid_factored_experiment import AdvancedMinigridFactoredExperiment
import argparse 
from experiments.minigrid.advanced_doorkey.core.policy_train_wrapper import AdvancedDoorKeyPolicyTrainWrapper
from portable.utils.utils import load_gin_configs
import torch
import random
from experiments.minigrid.advanced_doorkey.advanced_minigrid_option_resources import *

import matplotlib.pyplot as plt

positive_files = [
    "resources/minigrid_factored/adv_doorkey_8x8_openreddoor_doorred_4_initiation_positive.npy",
    # "resources/minigrid_factored/adv_doorkey_8x8_openreddoor_doorred_1_initiation_positive.npy",
    "resources/minigrid_factored/adv_doorkey_8x8_openreddoor_doorred_2_initiation_positive.npy",
    "resources/minigrid_factored/adv_doorkey_8x8_openreddoor_doorred_3_initiation_positive.npy",
]

negative_files = [
    "resources/minigrid_factored/adv_doorkey_8x8_openreddoor_doorred_4_initiation_negative.npy",
    # "resources/minigrid_factored/adv_doorkey_8x8_openreddoor_doorred_1_initiation_negative.npy",
    "resources/minigrid_factored/adv_doorkey_8x8_openreddoor_doorred_2_initiation_negative.npy",
    "resources/minigrid_factored/adv_doorkey_8x8_openreddoor_doorred_3_initiation_negative.npy",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config_file", nargs='+', type=str, required=True)
    parser.add_argument("--gin_bindings", default=[], help='Gin bindings to override the values' + 
            ' set in the config files (e.g. "DQNAgent.epsilon_train=0.1",' +
            ' "create_atari_environment.game_name="Pong"").')
    
    args = parser.parse_args()
    
    load_gin_configs(args.config_file, args.gin_bindings)
    
    def policy_phi(x):
        return x
    
    experiment = AdvancedMinigridFactoredExperiment(base_dir=args.base_dir,
                                                    training_seed=training_seed,
                                                    experiment_seed=args.seed,
                                                    policy_phi=policy_phi,
                                                    termination_oracles=check_got_redkey,
                                                    markov_option_builder=None)
    
    # for idx in range(len(positive_files)):
    for idx in range(1):
        
        train_positive = positive_files[:idx+1]
        train_negative = negative_files[:idx+1]
        
        test_positive = positive_files[idx+1:]
        test_negative = negative_files[idx+1:]
        
        experiment.add_datafiles(train_positive,
                                 train_negative)
        
        experiment.train_classifier(300)
        
        loss, accuracy = experiment.test_classifier(test_positive,
                                                    test_negative)
        
        print(loss)
        print(accuracy)
        
        experiment.reset_option()
    


