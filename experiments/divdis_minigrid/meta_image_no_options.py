from experiments.divdis_minigrid.core.advanced_minigrid_divdis_meta_experiment import AdvancedMinigridDivDisMetaExperiment
import argparse
from portable.utils.utils import load_gin_configs
import torch 
from experiments.minigrid.utils import environment_builder
from experiments.minigrid.advanced_doorkey.core.policy_train_wrapper import AdvancedDoorKeyPolicyTrainWrapper
import random
from experiments.divdis_minigrid.core.advanced_minigrid_mock_terminations import *
from portable.agent.model.ppo import CNNPolicy, CNNVF

env_seed = 1

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
    
    def option_agent_phi(x):        
        return x
    
    terminations = []
    
    experiment = AdvancedMinigridDivDisMetaExperiment(base_dir=args.base_dir,
                                                      seed=args.seed,
                                                      option_policy_phi=policy_phi,
                                                      agent_phi=option_agent_phi,
                                                      action_policy=CNNPolicy(3, 7),
                                                      action_vf=CNNVF(3),
                                                      terminations=terminations)
    
    # experiment.load()
    
    meta_env = environment_builder(
                    'AdvancedDoorKey-8x8-v0',
                    seed=env_seed,
                    max_steps=int(1e4),
                    grayscale=False
                )
    
    experiment.train_meta_agent(meta_env,
                                env_seed,
                                2e8,
                                0.7)
    
    
    
    
