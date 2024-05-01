from experiments.divdis_minigrid.core.advanced_minigrid_divdis_meta_experiment import FactoredAdvancedMinigridDivDisMetaExperiment
import argparse
from portable.utils.utils import load_gin_configs
import torch 
from experiments.minigrid.utils import factored_environment_builder
from experiments.minigrid.advanced_doorkey.core.policy_train_wrapper import AdvancedDoorKeyPolicyTrainWrapper
import random
from experiments.divdis_minigrid.core.advanced_minigrid_mock_terminations import *
from portable.agent.model.ppo import create_linear_policy, create_linear_vf



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
        x = x/torch.tensor([7,7,1,1,5,7,7,5,7,7,5,7,7,5,7,7,5,7,7,5,7,7,4,7,7,7])

        return x
    
    def option_agent_phi(x):
        x = x/torch.tensor([7,7,1,1,5,7,7,5,7,7,5,7,7,5,7,7,5,7,7,5,7,7,4,7,7,7,1,1,1,1,1,1,1])
        
        return x
    
    terminations = []
    
    experiment = FactoredAdvancedMinigridDivDisMetaExperiment(base_dir=args.base_dir,
                                                              seed=args.seed,
                                                              option_policy_phi=policy_phi,
                                                              option_agent_phi=option_agent_phi,
                                                              action_policy=create_linear_policy(26, 7),
                                                              action_vf=create_linear_vf(26),
                                                              option_policy=create_linear_policy(33, 1),
                                                              option_vf=create_linear_vf(33),
                                                              terminations=terminations)
    
    # experiment.load()
    
    meta_env = factored_environment_builder(
                    'AdvancedDoorKey-8x8-v0',
                    seed=env_seed,
                    max_steps=int(1e4)
                )
    
    experiment.train_meta_agent(meta_env,
                                env_seed,
                                2e8,
                                0.7)
    
    
    
    
