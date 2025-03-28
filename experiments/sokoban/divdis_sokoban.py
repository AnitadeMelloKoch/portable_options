from experiments.sokoban.utils import environment_builder
from experiments.core.divdis_meta_masked_ppo_experiment import DivDisMetaMaskedPPOExperiment
import argparse 
from portable.utils.utils import load_gin_configs
from portable.agent.model.ppo import create_cnn_policy, create_cnn_vf
from experiments.sokoban.experiment_files import *
import numpy as np
import torch


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
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        x = (x/255.0).float()
        
        # may need to permute channels
        
        return x
    
    def option_agent_phi(x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        x = (x/255.0).float()
        
        # may need to permute channels
        
        return x
    
    def termination_phi(x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        x = (x/255.0).float()
        
        # may need to permute channels
        
        return x
    
    experiment = DivDisMetaMaskedPPOExperiment()
    
    experiment.add_datafiles(soko_positive_files,
                             soko_negative_files,
                             soko_unlabelled_files)
    
    experiment.train_option_classifiers()
    
    experiment.test_classifiers(soko_test_positive,
                                soko_test_negative)

    env = environment_builder(seed=args.seed)
    
    




