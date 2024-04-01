from experiments.divdis_minigrid.core.advanced_minigrid_divdis_meta_experiment import AdvancedMinigridDivDisMetaExperiment
import argparse
from portable.utils.utils import load_gin_configs
import torch 
from experiments.minigrid.utils import environment_builder
from experiments.minigrid.advanced_doorkey.core.policy_train_wrapper import AdvancedDoorKeyPolicyTrainWrapper
import random
from experiments.divdis_minigrid.core.advanced_minigrid_mock_terminations import *
from portable.agent.model.ppo import create_cnn_policy, create_cnn_vf

def make_random_getkey_env(train_colour, seed, collect_key=False):
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
        time_limit=100,
        image_input=False,
        key_collected=collect_key,
        keep_colour=train_colour
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
        time_limit=100,
        image_input=False,
        keep_colour="red"
        )],
    ],
    [
        [AdvancedDoorKeyPolicyTrainWrapper(
        environment_builder(
            'AdvancedDoorKey-8x8-v0',
            seed=env_seed,
            grayscale=False
        ),
        door_colour="red",
        time_limit=100,
        image_input=False,
        keep_colour="yellow"
        )],
    ],
    [
        [AdvancedDoorKeyPolicyTrainWrapper(
        environment_builder(
            'AdvancedDoorKey-8x8-v0',
            seed=env_seed,
            grayscale=False
        ),
        door_colour="red",
        time_limit=100,
        image_input=False,
        keep_colour="grey"
        )],
    ],
    [
        [AdvancedDoorKeyPolicyTrainWrapper(
        environment_builder(
            'AdvancedDoorKey-8x8-v0',
            seed=env_seed,
            grayscale=False
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
            environment_builder(
                'AdvancedDoorKey-8x8-v0',
                seed=env_seed,
                grayscale=False
            ),
            door_colour="red",
            time_limit=300,
            image_input=False,
            force_door_open=True
        )]
    ],
    [
        [AdvancedDoorKeyPolicyTrainWrapper(
            environment_builder(
                'AdvancedDoorKey-8x8-v0',
                seed=env_seed,
                grayscale=False
            ),
            door_colour="red",
            time_limit=300,
            image_input=False,
            force_door_open=True
        )]
    ],
    [
        [AdvancedDoorKeyPolicyTrainWrapper(
            environment_builder(
                'AdvancedDoorKey-8x8-v0',
                seed=env_seed,
                grayscale=False
            ),
            door_colour="red",
            time_limit=300,
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
        if type(x) is np.ndarray:
            if np.max(x) > 1:
                x = x/255.0
            x = x.astype(np.float32)
        else:
            if torch.max(x) > 1:
                x = x/255.0
        return x
    
    def option_agent_phi(x):
        return x
    
    terminations = [
        [PerfectGetKey("red"),
         NeverCorrectGetKey("red")],
        [PerfectGetKey("yellow"),
         NeverCorrectGetKey("yellow")],
        [PerfectGetKey("grey"),
         NeverCorrectGetKey("grey")],
        [PerfectDoorOpen(),
         PerfectDoorOpen()],
        [PerfectAtLocation(4,1),
         PerfectAtLocation(4,1)],
        [PerfectAtLocation(5,3),
         PerfectAtLocation(5,3)],
        [PerfectAtLocation(6,6),
         PerfectAtLocation(6,6)],
    ]
    
    experiment = AdvancedMinigridDivDisMetaExperiment(base_dir=args.base_dir,
                                                              seed=args.seed,
                                                              option_policy_phi=policy_phi,
                                                              option_agent_phi=option_agent_phi,
                                                              action_policy=create_cnn_policy(3, 14),
                                                              action_vf=create_cnn_vf(3),
                                                              option_policy=create_cnn_policy(3, 3),
                                                              option_vf=create_cnn_vf(3),
                                                              terminations=terminations)
    
    
    # experiment.train_option_policies(train_envs,
    #                                  env_seed,
    #                                  4e6)
    
    experiment.load()
    
    meta_env = environment_builder(
                    'AdvancedDoorKey-8x8-v0',
                    seed=env_seed,
                    max_steps=int(1e4),
                    grayscale=False
                )
    
    experiment.train_meta_agent(meta_env,
                                env_seed,
                                2e8,
                                2)
    
    experiment.eval_meta_agent(meta_env,
                               env_seed,
                               1)
    
    
