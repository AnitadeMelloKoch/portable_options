from experiments.divdis_minigrid.core.advanced_minigrid_factored_divdis_meta_experiment import FactoredAdvancedMinigridDivDisMetaExperiment
import argparse
from portable.utils.utils import load_gin_configs
import torch 
from experiments.minigrid.utils import factored_environment_builder
from experiments.minigrid.advanced_doorkey.core.policy_train_wrapper import AdvancedDoorKeyPolicyTrainWrapper
import random
from experiments.divdis_minigrid.core.advanced_minigrid_mock_terminations import *
from portable.agent.model.linear_q import *

def make_random_getkey_env(train_colour, seed, collect_key=False):
    colours = ["red", "green", "blue", "purple", "yellow", "grey"]
    possible_key_colours = list(filter(lambda c: c!= train_colour, colours))
    
    door_colour = random.choice(possible_key_colours)
    possible_key_colours = list(filter(lambda c: c!= door_colour, possible_key_colours))
    random.shuffle(possible_key_colours)
    key_cols = [train_colour] + possible_key_colours
    
    return AdvancedDoorKeyPolicyTrainWrapper(
        factored_environment_builder(
            'AdvancedDoorKey-8x8-v0',
            seed=seed,
        ),
        door_colour=door_colour,
        key_colours=key_cols,
        time_limit=100,
        image_input=False,
        key_collected=collect_key
    )

env_seed = 1

train_envs = [
    [
        [AdvancedDoorKeyPolicyTrainWrapper(
        factored_environment_builder(
            'AdvancedDoorKey-8x8-v0',
            seed=env_seed
        ),
        door_colour="blue",
        time_limit=500,
        image_input=False
        ),
        make_random_getkey_env("blue", env_seed),
        make_random_getkey_env("blue", env_seed),
        make_random_getkey_env("blue", env_seed),],
        [AdvancedDoorKeyPolicyTrainWrapper(
        factored_environment_builder(
            'AdvancedDoorKey-8x8-v0',
            seed=env_seed
        ),
        door_colour="blue",
        time_limit=500,
        image_input=False,
        key_collected=True
        ),
        make_random_getkey_env("blue", env_seed, True),
        make_random_getkey_env("blue", env_seed, True),
        make_random_getkey_env("blue", env_seed, True),],
    ],
    [
        [AdvancedDoorKeyPolicyTrainWrapper(
        factored_environment_builder(
            'AdvancedDoorKey-8x8-v0',
            seed=env_seed
        ),
        door_colour="yellow",
        time_limit=500,
        image_input=False
        ),
        make_random_getkey_env("yellow", env_seed),
        make_random_getkey_env("yellow", env_seed),
        make_random_getkey_env("yellow", env_seed),],
        [AdvancedDoorKeyPolicyTrainWrapper(
        factored_environment_builder(
            'AdvancedDoorKey-8x8-v0',
            seed=env_seed
        ),
        door_colour="yellow",
        time_limit=500,
        image_input=False,
        key_collected=True
        ),
        make_random_getkey_env("yellow", env_seed, True),
        make_random_getkey_env("yellow", env_seed, True),
        make_random_getkey_env("yellow", env_seed, True),],
    ],
    [
        [AdvancedDoorKeyPolicyTrainWrapper(
        factored_environment_builder(
            'AdvancedDoorKey-8x8-v0',
            seed=env_seed
        ),
        door_colour="grey",
        time_limit=500,
        image_input=False
        ),
        make_random_getkey_env("grey", env_seed),
        make_random_getkey_env("grey", env_seed),
        make_random_getkey_env("grey", env_seed),],
        [AdvancedDoorKeyPolicyTrainWrapper(
        factored_environment_builder(
            'AdvancedDoorKey-8x8-v0',
            seed=env_seed
        ),
        door_colour="grey",
        time_limit=500,
        image_input=False,
        key_collected=True
        ),
        make_random_getkey_env("grey", env_seed, True),
        make_random_getkey_env("grey", env_seed, True),
        make_random_getkey_env("grey", env_seed, True),],
    ]
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
        x = x/torch.tensor([7,7,1,1,5,7,7,5,7,7,5,7,7,5,7,7,5,7,7,5,7,7,4,7,7,7])

        return x
    
    action_agent = LargeLinearQFunction()
    option_agent = OptionLinearQFunction()
    terminations = [
        [PerfectGetKey("blue"),
         NeverCorrectGetKey("blue")],
        [PerfectGetKey("yellow"),
         NeverCorrectGetKey("yellow")],
        [PerfectGetKey("grey"),
         NeverCorrectGetKey("grey")]
    ]
    
    experiment = FactoredAdvancedMinigridDivDisMetaExperiment(base_dir=args.base_dir,
                                                              seed=args.seed,
                                                              meta_policy_phi=policy_phi,
                                                              action_agent=action_agent,
                                                              option_agent=option_agent,
                                                              terminations=terminations)
    
    experiment.train_option_policies(train_envs,
                                     env_seed,
                                     1e6)
    
    meta_env = AdvancedDoorKeyPolicyTrainWrapper(
        factored_environment_builder(
            'AdvancedDoorKey-8x8-v0',
            seed=env_seed
        ),
        door_colour="blue",
        time_limit=500,
        image_input=False
    )
    
    experiment.train_meta_agent(meta_env,
                                env_seed,
                                2e6,
                                0.7)
    
    
    
    
