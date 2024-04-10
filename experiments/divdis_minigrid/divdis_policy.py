import argparse
from portable.utils.utils import load_gin_configs
import torch 
from experiments.divdis_minigrid.core.advanced_minigrid_mock_terminations import *
from experiments.minigrid.advanced_doorkey.core.policy_train_wrapper import AdvancedDoorKeyPolicyTrainWrapper
from experiments.divdis_minigrid.core.advanced_minigrid_policy_experiment import AdvancedMinigridDivDisOptionExperiment
from experiments.minigrid.utils import environment_builder

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
    
    terminations = [
        [PerfectGetKey("red")],
        [PerfectGetKey("red")]
    ]
    
    env_1 = AdvancedDoorKeyPolicyTrainWrapper(environment_builder(
        'AdvancedDoorKey-8x8-v0',
        seed=0,
        max_steps=500,
        grayscale=False
        ),
        door_colour="red",
        time_limit=100,
        image_input=True
        )
    
    env_2 = AdvancedDoorKeyPolicyTrainWrapper(environment_builder(
        'AdvancedDoorKey-8x8-v0',
        seed=1,
        max_steps=500,
        grayscale=False
        ),
        door_colour="red",
        time_limit=100,
        image_input=True)
    
    experiment = AdvancedMinigridDivDisOptionExperiment(base_dir=args.base_dir,
                                                        seed=args.seed,
                                                        policy_phi=policy_phi,
                                                        option_type="mock")
    
    experiment.evaluate_diff_policies_mock_option(env_1,
                                                  env_2,
                                                  0,
                                                  1,
                                                  terminations,
                                                  "wass",
                                                  5)
    
    
    











