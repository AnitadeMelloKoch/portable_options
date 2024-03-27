from experiments.divdis_minigrid.core.advanced_minigrid_divdis_meta_experiment import AdvancedMinigridDivDisMetaExperiment
import argparse
from portable.utils.utils import load_gin_configs
import torch 
from experiments.minigrid.utils import environment_builder
from experiments.minigrid.advanced_doorkey.core.policy_train_wrapper import AdvancedDoorKeyPolicyTrainWrapper
import random
from experiments.divdis_minigrid.core.advanced_minigrid_mock_terminations import *
from portable.agent.model.ppo import create_linear_policy, create_linear_vf

def make_random_getkey_env(train_colour, 
                           seed, 
                           collect_key=False, 
                           keep_colour="",
                           pickup_colour=""):
    colours = ["red", "green", "blue", "purple", "yellow", "grey"]
    possible_key_colours = list(filter(lambda c: c!= train_colour, colours))
    
    door_colour = random.choice(possible_key_colours)
    possible_key_colours = list(filter(lambda c: c!= door_colour, possible_key_colours))
    random.shuffle(possible_key_colours)
    key_cols = [train_colour] + possible_key_colours
    
    return AdvancedDoorKeyPolicyTrainWrapper(
        environment_builder(
            'AdvancedDoorKey-8x8-v0',
            seed=seed,
            grayscale=False
        ),
        door_colour=door_colour,
        key_colours=key_cols,
        time_limit=500,
        image_input=False,
        key_collected=collect_key,
        keep_colour=keep_colour,
        pickup_colour=pickup_colour
    )

env_seed = 1

train_envs = [
    [
        [AdvancedDoorKeyPolicyTrainWrapper(
        environment_builder(
            'AdvancedDoorKey-8x8-v0',
            seed=env_seed,
            grayscale=False
        ),
        door_colour="red",
        time_limit=500,
        image_input=False,
        keep_colour="red"
        ),
        make_random_getkey_env("red", env_seed, keep_colour="red"),
        make_random_getkey_env("red", env_seed, keep_colour="red"),
        make_random_getkey_env("red", env_seed, keep_colour="red"),],
        [AdvancedDoorKeyPolicyTrainWrapper(
        environment_builder(
            'AdvancedDoorKey-8x8-v0',
            seed=env_seed,
            grayscale=False
        ),
        door_colour="red",
        time_limit=500,
        image_input=False,
        pickup_colour="red"
        ),
        make_random_getkey_env("red", env_seed, pickup_colour="red"),
        make_random_getkey_env("red", env_seed, pickup_colour="red"),
        make_random_getkey_env("red", env_seed, pickup_colour="red"),],
    ],
    [
        [AdvancedDoorKeyPolicyTrainWrapper(
        environment_builder(
            'AdvancedDoorKey-8x8-v0',
            seed=env_seed,
            grayscale=False
        ),
        door_colour="red",
        time_limit=500,
        image_input=False,
        keep_colour="yellow"
        ),
        make_random_getkey_env("yellow", env_seed, keep_colour="yellow"),
        make_random_getkey_env("yellow", env_seed, keep_colour="yellow"),
        make_random_getkey_env("yellow", env_seed, keep_colour="yellow"),],
        [AdvancedDoorKeyPolicyTrainWrapper(
        environment_builder(
            'AdvancedDoorKey-8x8-v0',
            seed=env_seed,
            grayscale=False
        ),
        door_colour="red",
        time_limit=500,
        image_input=False,
        pickup_colour="yellow"
        ),
        make_random_getkey_env("yellow", env_seed, pickup_colour="yellow"),
        make_random_getkey_env("yellow", env_seed, pickup_colour="yellow"),
        make_random_getkey_env("yellow", env_seed, pickup_colour="yellow"),],
    ],
    [
        [AdvancedDoorKeyPolicyTrainWrapper(
        environment_builder(
            'AdvancedDoorKey-8x8-v0',
            seed=env_seed,
            grayscale=False
        ),
        door_colour="red",
        time_limit=500,
        image_input=False,
        keep_colour="grey"
        ),
        make_random_getkey_env("grey", env_seed, keep_colour="grey"),
        make_random_getkey_env("grey", env_seed, keep_colour="grey"),
        make_random_getkey_env("grey", env_seed, keep_colour="grey"),],
        [AdvancedDoorKeyPolicyTrainWrapper(
        environment_builder(
            'AdvancedDoorKey-8x8-v0',
            seed=env_seed,
            grayscale=False
        ),
        door_colour="red",
        time_limit=500,
        image_input=False,
        pickup_colour="grey"
        ),
        make_random_getkey_env("grey", env_seed, pickup_colour="grey"),
        make_random_getkey_env("grey", env_seed, pickup_colour="grey"),
        make_random_getkey_env("grey", env_seed, pickup_colour="grey"),],
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
    
    def option_agent_phi(x):
        x = x/torch.tensor([7,7,1,1,5,7,7,5,7,7,5,7,7,5,7,7,5,7,7,5,7,7,4,7,7,7,1,1,1,1,1,1,1,1,1,1])
        
        return x
    
    terminations = [
        [PerfectGetKey("red"),
         NeverCorrectGetKey("red")],
        [PerfectGetKey("yellow"),
         NeverCorrectGetKey("yellow")],
        [PerfectGetKey("grey"),
         NeverCorrectGetKey("grey")]
    ]
    
    experiment = AdvancedMinigridDivDisMetaExperiment(base_dir=args.base_dir,
                                                              seed=args.seed,
                                                              option_policy_phi=policy_phi,
                                                              option_agent_phi=option_agent_phi,
                                                              action_policy=create_linear_policy(26, 10),
                                                              action_vf=create_linear_vf(26),
                                                              option_policy=create_linear_policy(36, 2),
                                                              option_vf=create_linear_vf(36),
                                                              terminations=terminations)
    
    # experiment.load()
    
    experiment.train_option_policies(train_envs,
                                     env_seed,
                                     1e7)
    
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
    
    
    
    
