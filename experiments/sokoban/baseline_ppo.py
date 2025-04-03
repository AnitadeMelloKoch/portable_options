from experiments.core.divdis_meta_masked_ppo_experiment_baseline import DivDisMetaMaskedPPOExperiment
import argparse 
from portable.utils.utils import load_gin_configs
from experiments.sokoban.utils import environment_builder
from portable.agent.model.ppo import create_cnn_policy, create_cnn_vf
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

    experiment = DivDisMetaMaskedPPOExperiment(base_dir=args.base_dir,
                                               seed=args.seed,
                                               option_policy_phi=policy_phi,
                                               agent_phi=option_agent_phi,
                                               action_policy=create_cnn_policy(3,13),
                                               action_vf=create_cnn_vf(3),
                                               option_type="mock")
    
    env = environment_builder(level_name="PushAndPull-Sokoban-v0",
                              max_steps=150, 
                              seed=args.seed,
                              scale_dims=(128, 128))
    
    experiment.train_meta_agent(env,
                                args.seed,
                                10e6,
                                0.98)