from experiments.divdis_minigrid.core.advanced_minigrid_factored_divdis_meta_experiment import FactoredAdvancedMinigridDivDisMetaExperiment
import argparse
from portable.utils.utils import load_gin_configs
import torch 
from experiments.minigrid.utils import factored_environment_builder
from experiments.minigrid.advanced_doorkey.core.policy_train_wrapper import AdvancedDoorKeyPolicyTrainWrapper
import random
from experiments.divdis_minigrid.core.advanced_minigrid_mock_terminations import *
from portable.agent.model.ppo import create_linear_policy, create_linear_vf

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
        key_collected=collect_key,
        keep_colour=train_colour
    )

env_seed = 1

train_envs = [
    [
        [AdvancedDoorKeyPolicyTrainWrapper(
        factored_environment_builder(
            'AdvancedDoorKey-8x8-v0',
            seed=env_seed
        ),
        door_colour="red",
        time_limit=100,
        image_input=False,
        keep_colour="red"
        )],
    ],
    [
        [AdvancedDoorKeyPolicyTrainWrapper(
        factored_environment_builder(
            'AdvancedDoorKey-8x8-v0',
            seed=env_seed
        ),
        door_colour="red",
        time_limit=100,
        image_input=False,
        keep_colour="yellow"
        )],
    ],
    [
        [AdvancedDoorKeyPolicyTrainWrapper(
        factored_environment_builder(
            'AdvancedDoorKey-8x8-v0',
            seed=env_seed
        ),
        door_colour="red",
        time_limit=100,
        image_input=False,
        keep_colour="grey"
        )],
    ],
    [
        [AdvancedDoorKeyPolicyTrainWrapper(
        factored_environment_builder(
            'AdvancedDoorKey-8x8-v0',
            seed=env_seed
        ),
        door_colour="red",
        time_limit=100,
        image_input=False,
        pickup_colour="red",
        force_door_closed=True
        )],
    ],
    [
        [AdvancedDoorKeyPolicyTrainWrapper(
            factored_environment_builder(
                'AdvancedDoorKey-8x8-v0',
                seed=env_seed
            ),
            door_colour="red",
            time_limit=100,
            image_input=False,
            force_door_open=True
        )]
    ],
    [
        [AdvancedDoorKeyPolicyTrainWrapper(
            factored_environment_builder(
                'AdvancedDoorKey-8x8-v0',
                seed=env_seed
            ),
            door_colour="red",
            time_limit=100,
            image_input=False,
            force_door_open=True
        )]
    ],
    [
        [AdvancedDoorKeyPolicyTrainWrapper(
            factored_environment_builder(
                'AdvancedDoorKey-8x8-v0',
                seed=env_seed
            ),
            door_colour="red",
            time_limit=100,
            image_input=False,
            force_door_open=True
        )]
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
        x = x/torch.tensor([7,7,1,1,5,7,7,5,7,7,5,7,7,5,7,7,5,7,7,5,7,7,4,7,7,7,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
        
        return x
    
    terminations = [
        [PerfectGetKey("red")],
        [PerfectGetKey("yellow")],
        [PerfectGetKey("grey")],
        [PerfectDoorOpen()],
        [PerfectAtLocation(4,4)],
        [PerfectAtLocation(5,5)],
        [PerfectAtLocation(6,6)],
    ]
    
    experiment = FactoredAdvancedMinigridDivDisMetaExperiment(base_dir=args.base_dir,
                                                              seed=args.seed,
                                                              option_policy_phi=policy_phi,
                                                              option_agent_phi=option_agent_phi,
                                                              action_policy=create_linear_policy(26, 14),
                                                              action_vf=create_linear_vf(26),
                                                              option_policy=create_linear_policy(40, 1),
                                                              option_vf=create_linear_vf(40),
                                                              terminations=terminations)
    
    
    experiment.train_option_policies(train_envs,
                                     env_seed,
                                     4e6)
    
    meta_env = factored_environment_builder(
                    'AdvancedDoorKey-8x8-v0',
                    seed=env_seed,
                    max_steps=int(1e4)
                )
    
    experiment.train_meta_agent(meta_env,
                                env_seed,
                                2e8,
                                0.7)
    
    experiment.eval_meta_agent(meta_env,
                               env_seed,
                               1)
    
    
