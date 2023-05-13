from experiments.minigrid.doorkey.core.minigrid_experiment import MinigridExperiment
from experiments.minigrid.doorkey.core.agents.minigrid_agent import MinigridAgentWrapper
import os
import numpy as np 
from experiments.minigrid.utils import environment_builder
import argparse 

from portable.utils.utils import load_gin_configs
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
    
    def create_agent(save_dir, device, training_env, preprocess_obs):
        return MinigridAgentWrapper(save_dir=save_dir, 
                                    device=device,
                                    training_env=training_env,
                                    preprocess_obs=preprocess_obs)
    
    def create_env(seed):
        return environment_builder('MiniGrid-DoorKey-16x16-v0', 
                                   seed=seed,
                                   grayscale=False)
    
    def preprocess_obs(x, device):
        x = np.array(x)
        x = torch.tensor(x, device=device)
        x = x/255.0
        
        return x
    
    experiment = MinigridExperiment(base_dir=args.base_dir,
                            random_seed=args.seed,
                            create_env_function=create_env,
                            create_agent_function=create_agent,
                            preprocess_obs=preprocess_obs
                            )
    
    experiment.train()
    
    


