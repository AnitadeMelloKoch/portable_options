import argparse 
import logging 
import multiprocessing
import random
import warnings 

import numpy as np 

from experiments.divdis_monte.core.divdis_monte_classifier_experiment import MonteDivDisClassifierExperiment
from portable.utils.utils import load_gin_configs

positive_train_files = [
    ["resources/monte_images/lasers_left_toleft_room0_termination_positive.npy"],
    ["resources/monte_images/lasers_right_toleft_room0_termination_positive.npy"],
    ["resources/monte_images/lasers_left_toleft_room7_termination_positive.npy"],
    ["resources/monte_images/lasers_right_toleft_room7_termination_positive.npy"],
    ["resources/monte_images/lasers_toleft_room12_termination_positive.npy"],
]
negative_train_files = [
    ["resources/monte_images/lasers_left_toleft_room0_termination_negative.npy"],
    ["resources/monte_images/lasers_right_toleft_room0_termination_negative.npy"],
    ["resources/monte_images/lasers_left_toleft_room7_termination_negative.npy"],
    ["resources/monte_images/lasers_right_toleft_room7_termination_negative.npy"],
    ["resources/monte_images/lasers_toleft_room12_termination_negative.npy",
     "resources/monte_images/lasers_death_fromleft_room12_termination_negative.npy"],
]
unlabelled_train_files = [
    ["resources/monte_images/lasers_right_toleft_room0_termination_positive.npy",
     "resources/monte_images/lasers_right_toleft_room0_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room2_initiation_positive.npy",
     "resources/monte_images/lasers_right_toright_room7_termination_positive.npy",
     "resources/monte_images/lasers_right_toright_room7_termination_negative.npy",],
    ["resources/monte_images/climb_down_ladder_room2_initiation_positive.npy",],
    ["resources/monte_images/lasers_toright_room12_termination_negative.npy",
     "resources/monte_images/lasers_toright_room12_termination_positive.npy"],
    ["resources/monte_images/climb_down_ladder_room0_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room0_1_termination_negative.npy",],
    ["resources/monte_images/climb_down_ladder_room22_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room22_1_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room22_termination_positive.npy",
     "resources/monte_images/climb_down_ladder_room22_1_termination_positive.npy"]
]

positive_test_files = [
    "resources/monte_images/lasers_left_toleft_room0_termination_positive.npy",
    "resources/monte_images/lasers_right_toleft_room0_termination_positive.npy",
    "resources/monte_images/lasers_left_toleft_room7_termination_positive.npy",
    "resources/monte_images/lasers_right_toleft_room7_termination_positive.npy",
    "resources/monte_images/lasers_toleft_room12_termination_positive.npy",
]

negative_test_files = [
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
        parser.add_argument("--seed", type=int, required=True)
        parser.add_argument("--config_file", nargs='+', type=str, required=True)
        parser.add_argument("--gin_bindings", default=[], help='Gin bindings to override the values' + 
                ' set in the config files (e.g. "DQNAgent.epsilon_train=0.1",' +
                ' "create_atari_environment.game_name="Pong"").')

        args = parser.parse_args()

        load_gin_configs(args.config_file, args.gin_bindings)

        multiprocessing.set_start_method('spawn')
        #ignore warnings: UserWarning: Lazy modules are a new feature under heavy development
        warnings.filterwarnings("ignore", category=UserWarning, message="Lazy modules are a new feature under heavy development")

        experiment = MonteDivDisClassifierExperiment(base_dir=args.base_dir, seed=args.seed)


        experiment.add_test_files(positive_test_files,
                                  negative_test_files)

        experiment.room_by_room_train_labelled('laser', 'termination', 
                                               files_list=(positive_train_files, negative_train_files,
                                                           unlabelled_train_files),)
        
