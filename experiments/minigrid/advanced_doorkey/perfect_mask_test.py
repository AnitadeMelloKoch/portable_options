from experiments.minigrid.advanced_doorkey.core.advanced_minigrid_factored_experiment import AdvancedMinigridFactoredExperiment
import argparse 
from experiments.minigrid.advanced_doorkey.core.policy_train_wrapper import AdvancedDoorKeyPolicyTrainWrapper
from experiments.minigrid.utils import factored_environment_builder
from portable.utils.utils import load_gin_configs
import torch
import random
from experiments.minigrid.advanced_doorkey.advanced_minigrid_option_resources import *

import matplotlib.pyplot as plt 

import torch.nn as nn
from portable.option.ensemble.custom_attention import *

def make_random_getkey_env(train_colour, check_option_complete, seed):
    colours = ["red", "green", "blue", "purple", "yellow", "grey"]
    possible_key_colours = list(filter(lambda c: c!= train_colour, colours))
    
    door_colour = random.choice(possible_key_colours)
    possible_key_colours = list(filter(lambda c: c!= door_colour, possible_key_colours))
    other_col = random.choice(possible_key_colours)
    key_cols = [train_colour, other_col]
    random.shuffle(key_cols)
    
    return AdvancedDoorKeyPolicyTrainWrapper(
        factored_environment_builder(
            'AdvancedDoorKey-8x8-v0',
            seed=seed,
        ),
        check_option_complete=check_option_complete,
        door_colour=door_colour,
        key_colours=key_cols,
        time_limit=50,
        image_input=False
    )

def training_envs(seeds):
    print(seeds)
    training_envs = []
    for seed in seeds:
        training_envs.append(AdvancedDoorKeyPolicyTrainWrapper(
                factored_environment_builder(
                    'AdvancedDoorKey-8x8-v0',
                    seed=seed
                ),
                check_option_complete=check_got_redkey,
                door_colour="red",
                time_limit=50,
                image_input=False
            )
        )
        training_envs.append(make_random_getkey_env("red", check_got_redkey, seed))
        training_envs.append(make_random_getkey_env("red", check_got_redkey, seed))
        training_envs.append(make_random_getkey_env("red", check_got_redkey, seed))
        training_envs.append(make_random_getkey_env("red", check_got_redkey, seed))
        training_envs.append(make_random_getkey_env("red", check_got_redkey, seed))
    
    return training_envs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_envs", type=int, required=True)
    parser.add_argument("--config_file", nargs='+', type=str, required=True)
    parser.add_argument("--gin_bindings", default=[], help='Gin bindings to override the values' + 
            ' set in the config files (e.g. "DQNAgent.epsilon_train=0.1",' +
            ' "create_atari_environment.game_name="Pong"").')
    
    args = parser.parse_args()
    
    load_gin_configs(args.config_file, args.gin_bindings)
    
    def policy_phi(x):
        x = x/torch.tensor([7,7,1,5,7,7,7,7,7,7,7,7,7,7,7,7,7,7,4,7,7,7])

        return x
    
    experiment = AdvancedMinigridFactoredExperiment(base_dir=args.base_dir,
                                                    training_seed=training_seed,
                                                    experiment_seed=args.seed,
                                                    policy_phi=policy_phi,
                                                    termination_oracles=check_got_redkey,
                                                    markov_option_builder=None)
    
    for train_seed in range(20):
        experiment.reset_option()
        experiment.train_policy(training_envs(range(train_seed + 1)))
        for _ in range(args.num_envs):
            seed = random.randint(21, 1000)
            for idx in range(experiment.option.policy.num_modules):
                test_env = AdvancedDoorKeyPolicyTrainWrapper(
                    factored_environment_builder(
                        'AdvancedDoorKey-8x8-v0',
                        seed=seed
                    ),
                    check_option_complete=check_got_redkey,
                    door_colour="red",
                    time_limit=50,
                    image_input=False
                )
                experiment.run_episode(test_env,
                                       idx,
                                       "{}trains-seed{}policy{}".format(train_seed+1,seed, idx),
                                       train_seed+1)
    
