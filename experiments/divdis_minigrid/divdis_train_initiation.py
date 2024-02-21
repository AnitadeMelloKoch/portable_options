import argparse
from portable.utils.utils import load_gin_configs
import torch 
from experiments.minigrid.utils import factored_environment_builder
from experiments.minigrid.advanced_doorkey.core.policy_train_wrapper import AdvancedDoorKeyPolicyTrainWrapper
import random
from experiments.divdis_minigrid.core.advanced_minigrid_mock_terminations import *
from portable.agent.model.ppo import create_linear_policy, create_linear_vf
from experiments.divdis_minigrid.core.advanced_minigrid_factored_divdis_initiation_experiment import FactoredAdvancedMinigridDivDisInitiationExperiment
from experiments.divdis_minigrid.core.advanced_minigrid_mock_terminations import *

env_seed = 1

envs = [AdvancedDoorKeyPolicyTrainWrapper(
    factored_environment_builder(
        'AdvancedDoorKey-8x8-v0',
        seed=env_seed
    ),
    time_limit=1000,
    image_input=False,
    keep_colour="red"
)]

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
        
        terminations = [PerfectGetKey("red")]
        
        experiment = FactoredAdvancedMinigridDivDisInitiationExperiment(base_dir=args.base_dir,
                                                                        seed=args.seed,
                                                                        policy_phi=policy_phi,
                                                                        terminations=terminations)
        
        experiment.train(envs,
                         1e8)
        
        

