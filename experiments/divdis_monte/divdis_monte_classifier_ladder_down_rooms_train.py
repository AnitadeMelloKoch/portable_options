import logging
import multiprocessing
import os

import numpy as np
from experiments.divdis_monte.core.divdis_monte_classifier_experiment import MonteDivDisClassifierExperiment
import argparse 
from portable.utils.utils import load_gin_configs
import torch 
import random 

img_dir = "resources/monte_images/"

positive_files = [
    ["resources/monte_images/climb_down_ladder_room6_termination_positive.npy",
     "resources/monte_images/climb_down_ladder_room6_1_termination_positive.npy"],
    ["resources/monte_images/climb_down_ladder_room1_termination_positive.npy"],
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
    ["resources/monte_images/climb_down_ladder_room2_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room2_1_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room6_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room6_1_termination_negative.npy"],
    ["resources/monte_images/climb_down_ladder_room1_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room1_1_termination_negative.npy",
     "resources/monte_images/screen_death_1.npy",
     "resources/monte_images/screen_death_2.npy",
     "resources/monte_images/screen_death_3.npy",
     "resources/monte_images/screen_death_4.npy"],
    ["resources/monte_images/climb_down_ladder_room0_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room0_1_termination_negative.npy",
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
    ["resources/monte_images/lasers_toleft_room12_termination_positive.npy",
     "resources/monte_images/climb_down_ladder_room7_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room13_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room21_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room11_1_uncertain.npy",],
    ["resources/monte_images/climb_down_ladder_room9_termination_positive.npy",
     "resources/monte_images/climb_down_ladder_room9_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room10_uncertain.npy",
     "resources/monte_images/climb_down_ladder_room0_uncertain.npy",
     "resources/monte_images/room18_walk_around.npy",
     "resources/monte_images/lasers_left_toleft_room7_termination_negative.npy",
     "resources/monte_images/lasers_right_toleft_room7_termination_negative.npy",],
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

positive_test_files = [
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

negative_test_files = [
    "resources/monte_images/screen_climb_down_ladder_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room2_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room6_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room6_1_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room1_1_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room0_initiation_negative.npy",
    "resources/monte_images/climb_down_ladder_room2_1_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room4_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room4_1_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room10_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room10_1_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room3_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room3_1_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room9_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room9_1_termination_negative.npy",
    "resources/monte_images/climb_down_ladder_room0_1_termination_negative.npy",
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

uncertain_test_files = [img_dir+"climb_down_ladder_room0_uncertain.npy",
                        img_dir+"climb_down_ladder_room2_uncertain.npy",
                        img_dir+"climb_down_ladder_room3_uncertain.npy",
                        img_dir+"climb_down_ladder_room4_uncertain.npy",
                        img_dir+"climb_down_ladder_room5_uncertain.npy",
                        img_dir+"climb_down_ladder_room6_uncertain.npy",
                        img_dir+"climb_down_ladder_room7_uncertain.npy",
                        img_dir+"climb_down_ladder_room9_uncertain.npy",
                        img_dir+"climb_down_ladder_room10_uncertain.npy",
                        img_dir+"climb_down_ladder_room11_uncertain.npy",
                        img_dir+"climb_down_ladder_room13_uncertain.npy",
                        img_dir+"climb_down_ladder_room14_uncertain.npy",
                        img_dir+"climb_down_ladder_room19_uncertain.npy",
                        img_dir+"climb_down_ladder_room21_uncertain.npy",
                        img_dir+"climb_down_ladder_room22_uncertain.npy",

                        img_dir+"climb_down_ladder_room3_1_uncertain.npy",
                        img_dir+"climb_down_ladder_room4_1_uncertain.npy",
                        img_dir+"climb_down_ladder_room6_1_uncertain.npy",
                        img_dir+"climb_down_ladder_room9_1_uncertain.npy",
                        img_dir+"climb_down_ladder_room10_1_uncertain.npy",
                        img_dir+"climb_down_ladder_room11_1_uncertain.npy",
                        img_dir+"climb_down_ladder_room13_1_uncertain.npy",
                        img_dir+"climb_down_ladder_room19_1_uncertain.npy",
                        img_dir+"climb_down_ladder_room21_1_uncertain.npy",
                        img_dir+"climb_down_ladder_room22_1_uncertain.npy",
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

        #multiprocessing.set_start_method('spawn')

        experiment = MonteDivDisClassifierExperiment(base_dir=args.base_dir, seed=args.seed)


        experiment.add_test_files(positive_test_files,
                                  negative_test_files,
                                  uncertain_test_files)
        experiment.read_directory(img_dir)

        experiment.room_by_room_train_labelled('climb_down_ladder', 'termination', 
                                               files_list=(positive_files, negative_files,
                                                           unlabelled_files))
        
 