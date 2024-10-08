from experiments.core.divdis_option_experiment import DivDisOptionExperiment
import argparse 
from portable.utils.utils import load_gin_configs
import numpy as np
import torch
import os

from experiments.divdis_monte.core.monte_terminations import *
from experiments.divdis_monte.experiment_files import *

init_states = [
    ["resources/monte_env_states/room1/ladder/middle_top_0.pkl",
     "resources/monte_env_states/room1/ladder/left_top_0.pkl",
     "resources/monte_env_states/room1/ladder/right_top_0.pkl",],
    ["resources/monte_env_states/room0/ladder/top_0.pkl",
     "resources/monte_env_states/room0/ladder/top_1.pkl",
     "resources/monte_env_states/room0/ladder/top_2.pkl",
     "resources/monte_env_states/room0/ladder/top_3.pkl",],
    ["resources/monte_env_states/room2/ladder/top_0.pkl",
     "resources/monte_env_states/room2/ladder/top_1.pkl",
     "resources/monte_env_states/room2/ladder/top_2.pkl",],
    ["resources/monte_env_states/room3/ladder/top_0.pkl"],
    ["resources/monte_env_states/room5/ladder/top_0.pkl",
     "resources/monte_env_states/room5/ladder/top_1.pkl",
     "resources/monte_env_states/room5/ladder/top_2.pkl",],
    ["resources/monte_env_states/room7/ladder/top_0.pkl",
     "resources/monte_env_states/room7/ladder/top_1.pkl",
     "resources/monte_env_states/room7/ladder/top_2.pkl",],
    ["resources/monte_env_states/room14/ladder/top_0.pkl",
     "resources/monte_env_states/room14/ladder/top_1.pkl",
     "resources/monte_env_states/room14/ladder/top_2.pkl"]
]

term_points = [
    [
        [
            (76, 192, 1)
        ],
        [
            (20, 148, 1)
        ],
        [
            (133, 148, 1)
        ],
    ],
    [
        [
            (77,235,4),
            (77,235,10)
        ],
        [
            (77,235,4),
            (77,235,10)
        ],
        [
            (77,235,4),
            (77,235,10)
        ],
        [
            (77,235,4),
            (77,235,10)
        ],
    ],
    [
        [
            (77, 235, 6)
        ],
        [
            (77, 235, 6)
        ],[
            (77, 235, 6)
        ],
    ],
    [
        [ (77, 235, 9)]
    ],
    [
        [
            (77, 235, 11),
            (77, 235, 19)
        ],
        [
            (77, 235, 11),
            (77, 235, 19)
        ],
        [
            (77, 235, 11),
            (77, 235, 19)
        ],
    ],
    [
        [
            (77,235,13),
            (76, 235, 21)
        ],
        [
            (77,235,13),
            (76, 235, 21)
        ],
        [
            (77,235,13),
            (76, 235, 21)
        ]
    ],
    [
        [
            (77, 235, 22)
        ],
        [
            (77, 235, 22)
        ],
        [
            (77, 235, 22)
        ],
    ]
]

# files for each room
positive_files = [
    ["resources/monte_images/climb_down_ladder_room1_termination_positive.npy"],
    ["resources/monte_images/climb_down_ladder_room6_termination_positive.npy",
     "resources/monte_images/climb_down_ladder_room6_1_termination_positive.npy"],
    ["resources/monte_images/climb_down_ladder_room10_termination_positive.npy",
     "resources/monte_images/climb_down_ladder_room10_1_termination_positive.npy"],
    ["resources/monte_images/climb_down_ladder_room9_termination_positive.npy",
     "resources/monte_images/climb_down_ladder_room9_1_termination_positive.npy"],
    ["resources/monte_images/climb_down_ladder_room21_termination_positive.npy",
     "resources/monte_images/climb_down_ladder_room21_1_termination_positive.npy"],
    ["resources/monte_images/climb_down_ladder_room19_termination_positive.npy",
     "resources/monte_images/climb_down_ladder_room19_1_termination_positive.npy"],
    ["resources/monte_images/climb_down_ladder_room22_termination_positive.npy",
     "resources/monte_images/climb_down_ladder_room22_1_termination_positive.npy"]
]
negative_files = [
    ["resources/monte_images/climb_down_ladder_room1_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room1_extra_termination_negative.npy",
     "resources/monte_images/screen_death_1.npy",
     "resources/monte_images/screen_death_2.npy",
     "resources/monte_images/screen_death_3.npy",
     "resources/monte_images/screen_death_4.npy"],
    ["resources/monte_images/climb_down_ladder_room2_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room2_extra_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room6_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room6_1_termination_negative.npy"],
    ["resources/monte_images/climb_down_ladder_room0_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room0_extra_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room4_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room4_1_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room10_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room10_1_termination_negative.npy"],
    ["resources/monte_images/climb_down_ladder_room3_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room3_1_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room9_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room9_1_termination_negative.npy"],
    ["resources/monte_images/climb_down_ladder_room7_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room7_1_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room13_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room13_1_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room21_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room21_1_termination_negative.npy"],
    ["resources/monte_images/climb_down_ladder_room5_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room5_1_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room11_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room11_1_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room19_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room19_1_termination_negative.npy"],
    ["resources/monte_images/climb_down_ladder_room14_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room14_1_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room22_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room22_1_termination_negative.npy"],
]
unlabelled_files = [
    ["resources/monte_images/climb_down_ladder_room9_termination_positive.npy",
     "resources/monte_images/climb_down_ladder_room9_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room10_uncertain.npy",
     "resources/monte_images/climb_down_ladder_room0_uncertain.npy",
     "resources/monte_images/room18_walk_around.npy",
     "resources/monte_images/lasers_left_toleft_room7_termination_negative.npy",
     "resources/monte_images/lasers_right_toleft_room7_termination_negative.npy",],
    ["resources/monte_images/lasers_toleft_room12_termination_positive.npy",
     "resources/monte_images/climb_down_ladder_room7_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room13_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room21_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room11_1_uncertain.npy",],
    ["resources/monte_images/climb_down_ladder_room21_termination_positive.npy",
     "resources/monte_images/climb_down_ladder_room5_uncertain.npy",
     "resources/monte_images/climb_down_ladder_room13_uncertain.npy",
     "resources/monte_images/climb_down_ladder_room13_1_uncertain.npy",
     "resources/monte_images/climb_down_ladder_room4_uncertain.npy",
     "resources/monte_images/climb_down_ladder_room4_1_uncertain.npy",
     "resources/monte_images/climb_down_ladder_room7_uncertain.npy"],
    ["resources/monte_images/climb_down_ladder_room3_uncertain.npy",
     "resources/monte_images/climb_down_ladder_room3_1_uncertain.npy",
     "resources/monte_images/climb_down_ladder_room22_uncertain.npy",
     "resources/monte_images/climb_down_ladder_room22_1_uncertain.npy",
     "resources/monte_images/climb_down_ladder_room11_uncertain.npy",
     "resources/monte_images/climb_down_ladder_room9_uncertain.npy",
     "resources/monte_images/climb_down_ladder_room9_1_uncertain.npy",],
    ["resources/monte_images/climb_down_ladder_room2_uncertain.npy",
     "resources/monte_images/climb_down_ladder_room6_uncertain.npy",
     "resources/monte_images/climb_down_ladder_room6_1_uncertain.npy",],
    ["resources/monte_images/climb_down_ladder_room21_uncertain.npy",
     "resources/monte_images/climb_down_ladder_room21_1_uncertain.npy",
     "resources/monte_images/climb_down_ladder_room10_1_uncertain.npy",
     "resources/monte_images/climb_down_ladder_room10_uncertain.npy",],
    ["resources/monte_images/climb_down_ladder_room19_uncertain.npy",
     "resources/monte_images/climb_down_ladder_room19_1_uncertain.npy",]
]

test_files_positive = [
    "resources/monte_images/screen_climb_down_ladder_termination_positive.npy",
    "resources/monte_images/climb_down_ladder_room6_termination_positive.npy",
    "resources/monte_images/climb_down_ladder_room6_1_termination_positive.npy",
    "resources/monte_images/climb_down_ladder_room10_termination_positive.npy",
    "resources/monte_images/climb_down_ladder_room10_1_termination_positive.npy",
    "resources/monte_images/climb_down_ladder_room9_termination_positive.npy",
    "resources/monte_images/climb_down_ladder_room9_1_termination_positive.npy",
    "resources/monte_images/climb_down_ladder_room21_termination_positive.npy",
    "resources/monte_images/climb_down_ladder_room21_1_termination_positive.npy",
    "resources/monte_images/climb_down_ladder_room19_termination_positive.npy",
    "resources/monte_images/climb_down_ladder_room19_1_termination_positive.npy",
    "resources/monte_images/climb_down_ladder_room22_termination_positive.npy",
    "resources/monte_images/climb_down_ladder_room22_1_termination_positive.npy",
]

test_negative_files = [
    "resources/monte_images/screen_climb_down_ladder_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room2_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room6_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room6_1_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room1_extra_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room0_initiation_negative.npy",
    "resources/monte_images/climb_down_ladder_room2_extra_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room4_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room4_1_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room10_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room10_1_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room3_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room3_1_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room9_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room9_1_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room0_extra_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room7_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room13_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room13_1_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room21_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room21_1_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room5_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room5_1_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room11_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room11_1_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room19_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room19_1_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room14_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room14_1_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room22_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room22_1_termination_negative.npy"
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--sub_dir", type=str, default="")
    parser.add_argument("--seed", type=int, required=True)
    # number of rooms to train classifier
    parser.add_argument("--num_rooms", type=int, required=True)
    parser.add_argument("--config_file", nargs='+', type=str, required=True)
    parser.add_argument("--gin_bindings", default=[], help='Gin bindings to override the values' + 
            ' set in the config files (e.g. "DQNAgent.epsilon_train=0.1",' +
            ' "create_atari_environment.game_name="Pong"").')
    
    args = parser.parse_args()
    load_gin_configs(args.config_file, args.gin_bindings)

    def option_agent_phi(x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        x = (x/255.0).float()
        return x
    
    if args.sub_dir == "":
        base_dir = args.base_dir
    else:
        base_dir = os.path.join(args.base_dir, args.sub_dir)
    
    experiment = DivDisOptionExperiment(base_dir=base_dir,
                                        seed=args.seed,
                                        option_type="divdis",
                                        config_file=args.config_file,
                                        gin_bindings=args.gin_bindings)
    
    for file_idx in range(args.num_rooms):
        experiment.add_datafiles(positive_files[file_idx],
                                 negative_files[file_idx],
                                 unlabelled_files[file_idx])
    experiment.train_classifier()
    experiment.test_classifiers(test_files_positive,
                                test_negative_files)
    
    print("Classifiers trained. Starting policy training...")
    
    for state_idx, init_state in enumerate(init_states):
        print("Beginning training on env {} out of {}".format(state_idx, len(init_states)))
        experiment.change_option_save(name="room_idx_{}".format(state_idx))
        experiment.train_option(init_state,
                                term_points[state_idx],
                                args.seed,
                                3e5,
                                state_idx)
        experiment.save()


