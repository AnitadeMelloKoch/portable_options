from experiments.sokoban.utils import environment_builder
from experiments.core.divdis_meta_experiment import DivDisMetaExperiment
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
        
        x = x.permute((2,0,1))
        
        return x
    
    def option_agent_phi(x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        x = (x/255.0).float()
        
        x = x.permute((2,0,1))
                
        return x
    
    def termination_phi(x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        x = (x/255.0).float()
        
        if len(x.shape) == 3:
            x = x.permute(2,0,1)
        else:
            x = x.permute((0, 3, 1, 2))
                        
        return x
    
    experiment = DivDisMetaExperiment(base_dir=args.base_dir,
                                      seed=args.seed,
                                      option_policy_phi=policy_phi,
                                      agent_phi=option_agent_phi,
                                      termination_phi=termination_phi,
                                      action_policy=create_cnn_policy(3,15),
                                      action_vf=create_cnn_vf(3),
                                      option_type="divdis")
    
    experiment.add_datafiles(soko_positive_files,
                             soko_negative_files,
                             soko_unlabelled_files)
    
    experiment.train_option_classifiers()
    
    experiment.test_classifiers(soko_test_positive,
                                soko_test_negative)

    env = environment_builder(level_name="PushAndPull-Sokoban-v0",
                              max_steps=500, 
                              seed=args.seed,
                              scale_dims=(84, 84),
                              num_boxes=3)
    
    experiment.train_meta_agent(env,
                                args.seed,
                                1e7)
    
    




