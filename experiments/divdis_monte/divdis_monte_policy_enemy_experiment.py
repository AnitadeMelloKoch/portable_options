from experiments.core.divdis_option_experiment import DivDisOptionExperiment
import argparse 
from portable.utils.utils import load_gin_configs
import numpy as np
import torch
import os

from experiments.divdis_monte.core.monte_terminations import *
from experiments.divdis_monte.experiment_files import *



init_states = [
    ["resources/monte_env_states/room1/enemy/skull_right_0.pkl",
     "resources/monte_env_states/room1/enemy/skull_right_1.pkl",],
    ["resources/monte_env_states/room2/enemy/right_skull_0.pkl",
     "resources/monte_env_states/room2/enemy/right_skull_1.pkl",
     "resources/monte_env_states/room2/enemy/right_skull_2.pkl",],
    ["resources/monte_env_states/room3/enemy/right_of_skulls.pkl"],
    ["resources/monte_env_states/room4/enemy/right_of_spider_0.pkl",
     "resources/monte_env_states/room4/enemy/right_of_spider_1.pkl"],
    ["resources/monte_env_states/room5/enemy/right_of_skull_0.pkl"],
    ["resources/monte_env_states/room9/enemy/right_of_right_snake.pkl"],
    ["resources/monte_env_states/room11/enemy/right_of_right_snake.pkl",
     "resources/monte_env_states/room11/enemy/right_of_right_snake_2.pkl",
     "resources/monte_env_states/room11/enemy/right_of_right_snake_3.pkl"],
    ["resources/monte_env_states/room13/enemy/right_spider.pkl"],
    ["resources/monte_env_states/room18/enemy/right_skull.pkl"],
    ["resources/monte_env_states/room21/enemy/right_spider.pkl"],
    ["resources/monte_env_states/room22/enemy/right_snake.pkl"]
]

term_points = [
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
]

positive_files = [
    ["resources/monte_images/move_left_enemy_room1_termination_positive.npy"],
    ["resources/monte_images/move_left_enemy_room2_termination_positive.npy"],
    ["resources/monte_images/move_left_enemy_room3_termination_positive.npy"],
    ["resources/monte_images/move_left_enemy_room4_termination_positive.npy"],
    ["resources/monte_images/move_left_enemy_room5_termination_positive.npy"],
    ["resources/monte_images/move_left_enemy_room9right_termination_positive.npy"],
    ["resources/monte_images/move_left_enemy_room11right_termination_positive.npy"],
    ["resources/monte_images/move_left_enemy_room13_termination_positive.npy"],
    ["resources/monte_images/move_left_enemy_room18_termination_positive.npy"],
    ["resources/monte_images/move_left_enemy_room21_termination_positive.npy"],
    ["resources/monte_images/move_left_enemy_room22_termination_positive.npy"],
]

negative_files = [
    ["resources/monte_images/move_left_enemy_room1_termination_negative.npy"],
    ["resources/monte_images/move_left_enemy_room2_termination_negative.npy"],
    ["resources/monte_images/move_left_enemy_room3_termination_negative.npy"],
    ["resources/monte_images/move_left_enemy_room4_termination_negative.npy"],
    ["resources/monte_images/move_left_enemy_room5_termination_negative.npy"],
    ["resources/monte_images/move_left_enemy_room9left_termination_negative.npy"],
    ["resources/monte_images/move_left_enemy_room11left_termination_negative.npy"],
    ["resources/monte_images/move_left_enemy_room13_termination_negative.npy"],
    ["resources/monte_images/move_left_enemy_room18_termination_negative.npy"],
    ["resources/monte_images/move_left_enemy_room21_termination_negative.npy"],
    ["resources/monte_images/move_left_enemy_room22_termination_negative.npy"],
]

unlabelled_files = [
    ["resources/monte_images/move_left_enemy_room4_termination_positive.npy",
     "resources/monte_images/move_left_enemy_room4_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room0_initiation_positive.npy",
     "resources/monte_images/screen_death_3.npy",
     "resources/monte_images/screen_death_4.npy"],
    ["resources/monte_images/move_left_enemy_room9left_termination_negative.npy",
     "resources/monte_images/move_left_enemy_room9left_termination_positive.npy",
     "resources/monte_images/move_left_enemy_room5_termination_positive.npy",
     "resources/monte_images/climb_down_ladder_room19_termination_positive.npy",],
    ["resources/monte_images/move_left_enemy_room21_termination_positive.npy",
     "resources/monte_images/move_left_enemy_room21_termination_negative.npy"],
    ["resources/monte_images/room18_walk_around.npy",
     "resources/monte_images/move_left_enemy_room22_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room4_1_uncertain.npy"],
    ["resources/monte_images/climb_down_ladder_room9_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room9_1_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room11_termination_negative.npy",],
    ["resources/monte_images/climb_down_ladder_room21_uncertain.npy",
     "resources/monte_images/climb_down_ladder_room21_1_uncertain.npy",],
    ["resources/monte_images/climb_down_ladder_room2_extra_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room0_extra_termination_negative.npy",],
    ["resources/monte_images/move_left_enemy_room18_termination_negative.npy",
     "resources/monte_images/move_left_enemy_room18_termination_positive.npy"],
    ["resources/monte_images/climb_down_ladder_room21_1_uncertain.npy",
     "resources/monte_images/climb_down_ladder_room10_1_uncertain.npy",
     "resources/monte_images/climb_down_ladder_room10_uncertain.npy",],
    ["resources/monte_images/climb_down_ladder_room6_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room6_termination_positive.npy",],
    ["resources/monte_images/climb_down_ladder_room1_extra_termination_negative.npy",
     "resources/monte_images/screen_death_1.npy",
     "resources/monte_images/screen_death_2.npy",]
]

test_files_positive = [
    "resources/monte_images/move_left_enemy_room1_termination_positive.npy",
    "resources/monte_images/move_left_enemy_room2_termination_positive.npy",
    "resources/monte_images/move_left_enemy_room3_termination_positive.npy",
    "resources/monte_images/move_left_enemy_room4_termination_positive.npy",
    "resources/monte_images/move_left_enemy_room5_termination_positive.npy",
    "resources/monte_images/move_left_enemy_room9right_termination_positive.npy",
    "resources/monte_images/move_left_enemy_room11right_termination_positive.npy",
    "resources/monte_images/move_left_enemy_room13_termination_positive.npy",
    "resources/monte_images/move_left_enemy_room18_termination_positive.npy",
    "resources/monte_images/move_left_enemy_room21_termination_positive.npy",
    "resources/monte_images/move_left_enemy_room22_termination_positive.npy",
]

test_files_negative = [
    "resources/monte_images/move_left_enemy_room1_termination_negative.npy",
    "resources/monte_images/move_left_enemy_room2_termination_negative.npy",
    "resources/monte_images/move_left_enemy_room3_termination_negative.npy",
    "resources/monte_images/move_left_enemy_room4_termination_negative.npy",
    "resources/monte_images/move_left_enemy_room5_termination_negative.npy",
    "resources/monte_images/move_left_enemy_room9left_termination_negative.npy",
    "resources/monte_images/move_left_enemy_room11left_termination_negative.npy",
    "resources/monte_images/move_left_enemy_room13_termination_negative.npy",
    "resources/monte_images/move_left_enemy_room18_termination_negative.npy",
    "resources/monte_images/move_left_enemy_room21_termination_negative.npy",
    "resources/monte_images/move_left_enemy_room22_termination_negative.npy",
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
    
    if args.sub_dir == "":
        base_dir = args.base_dir
    else:
        base_dir = os.path.join(args.base_dir,
                                args.sub_dir)
    
    experiment = DivDisOptionExperiment(base_dir=base_dir,
                                        seed=args.seed,
                                        option_type="divdis",
                                        config_file=args.config_file,
                                        gin_bindings=args.gin_bindings,
                                        episode_life=True)
    
    for file_idx in range(args.num_rooms):
        experiment.add_datafiles(positive_files[file_idx],
                                 negative_files[file_idx],
                                 unlabelled_files[file_idx])
    experiment.train_classifier()
    experiment.test_classifiers(test_files_positive,
                                test_files_negative)
    
    print("Classifiers trained. Starting policy training...")
    
    for state_idx, init_state in enumerate(init_states):
        print("Beginning training on env {} out of {}".format(state_idx, len(init_states)))
        experiment.change_option_save(name="room_idx_{}".format(state_idx))
        experiment.train_option(init_state,
                                term_points[state_idx],
                                args.seed,
                                5e5,
                                state_idx)
        experiment.save()




