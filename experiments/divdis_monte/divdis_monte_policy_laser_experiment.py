from experiments.core.divdis_option_experiment import DivDisOptionExperiment
import argparse 
from portable.utils.utils import load_gin_configs
import numpy as np
import torch

from experiments.monte.environment import MonteBootstrapWrapper, MonteAgentWrapper
from portable.utils import load_init_states
from pfrl.wrappers import atari_wrappers
from experiments.divdis_monte.core.monte_terminations import *
from experiments.divdis_monte.experiment_files import *
import os

init_states = [
    ["resources/monte_env_states/room0/ladder/top_0.pkl"],
    ["resources/monte_env_states/room0/lasers/left_with_laser.pkl"],
    ["resources/monte_env_states/room7/ladder/top_0.pkl"],
    ["resources/monte_env_states/room7/lasers/between_right_lasers.pkl"],
    ["resources/monte_env_states/room12/platforms/right.pkl"]
]

term_points = [
    [
        [(5, 235, 0)],
    ],
    [
        [(100, 235, 0)]
    ],
    [
        [(20, 235, 7)]
    ],
    [
        [(103, 235, 7)]
    ]
]

# files for each room
positive_files = [
    ["resources/monte_images/lasers_left_toleft_room0_termination_positive.npy"],
    ["resources/monte_images/lasers_right_toleft_room0_termination_positive.npy"],
    ["resources/monte_images/lasers_left_toleft_room7_termination_positive.npy"],
    ["resources/monte_images/lasers_right_toleft_room7_termination_positive.npy"],
    ["resources/monte_images/lasers_toleft_room12_termination_positive.npy"],
]
negative_files = [
    ["resources/monte_images/lasers_left_toleft_room0_termination_negative.npy"],
    ["resources/monte_images/lasers_right_toleft_room0_termination_negative.npy"],
    ["resources/monte_images/lasers_left_toleft_room7_termination_negative.npy"],
    ["resources/monte_images/lasers_right_toleft_room7_termination_negative.npy"],
    ["resources/monte_images/lasers_toleft_room12_termination_negative.npy",
     "resources/monte_images/lasers_death_fromleft_room12_termination_negative.npy"],
]
unlabelled_files = [
    ["resources/monte_images/lasers_right_toleft_room0_termination_positive.npy",
     "resources/monte_images/lasers_right_toleft_room0_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room2_initiation_positive.npy",
     "resources/monte_images/lasers_right_toright_room7_termination_positive.npy",
     "resources/monte_images/lasers_right_toright_room7_termination_negative.npy",],
    ["resources/monte_images/climb_down_ladder_room2_initiation_positive.npy",],
    ["resources/monte_images/lasers_toright_room12_termination_negative.npy",
     "resources/monte_images/lasers_toright_room12_termination_positive.npy"],
    ["resources/monte_images/climb_down_ladder_room0_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room0_extra_termination_negative.npy",],
    ["resources/monte_images/climb_down_ladder_room22_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room22_1_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room22_termination_positive.npy",
     "resources/monte_images/climb_down_ladder_room22_1_termination_positive.npy"]
]

test_positive_files = [
    "resources/monte_images/lasers_left_toleft_room0_termination_positive.npy",
    "resources/monte_images/lasers_right_toleft_room0_termination_positive.npy",
    "resources/monte_images/lasers_left_toleft_room7_termination_positive.npy",
    "resources/monte_images/lasers_right_toleft_room7_termination_positive.npy",
    "resources/monte_images/lasers_toleft_room12_termination_positive.npy",
]

test_negative_files = [
    "resources/monte_images/lasers_left_toleft_room0_termination_negative.npy",
    "resources/monte_images/lasers_right_toleft_room0_termination_negative.npy",
    "resources/monte_images/lasers_left_toleft_room7_termination_negative.npy",
    "resources/monte_images/lasers_right_toleft_room7_termination_negative.npy",
    "resources/monte_images/lasers_toleft_room12_termination_negative.npy",
     "resources/monte_images/lasers_death_fromleft_room12_termination_negative.npy",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--sub_dir", type=str, default="")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_rooms", type=int, required=True)
    parser.add_argument("--config_file", nargs='+', type=str, required=True)
    parser.add_argument("--gin_bindings", default=[], help='Gin bindings to override the values' + 
            ' set in the config files (e.g. "DQNAgent.epsilon_train=0.1",' +
            ' "create_atari_environment.game_name="Pong"").')
    
    args = parser.parse_args()
    load_gin_configs(args.config_file, args.gin_bindings)
    
    if args.sub_dir == "":
        base_dir = args.base_dir
    else:
        base_dir = os.path.join(args.base_dir, args.sub_dir)
    
    def option_agent_phi(x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        x = (x/255.0).float()
        return x
    
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
    experiment.test_classifiers(test_positive_files,
                                test_negative_files)
    
    print("Classifiers trained. Starting policy training...")
    
    for state_idx, init_state in enumerate(init_states):
        print("Beginning training on env {} out of {}".format(state_idx, len(init_states)))
        experiment.change_option_save(name="room_idx_{}".format(state_idx))
        experiment.train_option(init_state,
                                term_points[state_idx],
                                args.seed,
                                2e6,
                                state_idx)
        experiment.save()


