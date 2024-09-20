import logging
import multiprocessing

import numpy as np
from experiments.divdis_monte.core.divdis_monte_classifier_experiment import MonteDivDisClassifierExperiment
import argparse 
from portable.utils.utils import load_gin_configs
import torch 
import random 

img_dir = "resources/monte_images/"

positive_test_files = [img_dir+"climb_down_ladder_room1_termination_positive.npy",
                       img_dir+"climb_down_ladder_room6_termination_positive.npy",
                       img_dir+"climb_down_ladder_room9_termination_positive.npy",
                       img_dir+"climb_down_ladder_room10_termination_positive.npy",
                       img_dir+"climb_down_ladder_room19_termination_positive.npy",
                       img_dir+"climb_down_ladder_room21_termination_positive.npy",
                       img_dir+"climb_down_ladder_room22_termination_positive.npy",
                       ]
negative_test_files = [img_dir+"climb_down_ladder_room1_termination_negative.npy",
                       img_dir+"screen_death_1.npy",
                       img_dir+"screen_death_2.npy",
                       img_dir+"screen_death_3.npy",
                       img_dir+"screen_death_4.npy",
                       img_dir+"climb_down_ladder_room2_termination_negative.npy",
                       img_dir+"climb_down_ladder_room3_termination_negative.npy",
                       img_dir+"climb_down_ladder_room4_termination_negative.npy",
                       img_dir+"climb_down_ladder_room5_termination_negative.npy",
                       img_dir+"climb_down_ladder_room6_termination_negative.npy",
                       img_dir+"climb_down_ladder_room7_termination_negative.npy",
                       img_dir+"climb_down_ladder_room9_termination_negative.npy",
                       img_dir+"climb_down_ladder_room10_termination_negative.npy",
                       img_dir+"climb_down_ladder_room11_termination_negative.npy",
                       img_dir+"climb_down_ladder_room13_termination_negative.npy",
                       img_dir+"climb_down_ladder_room14_termination_negative.npy",
                       img_dir+"climb_down_ladder_room19_termination_negative.npy",
                       img_dir+"climb_down_ladder_room21_termination_negative.npy",
                       img_dir+"climb_down_ladder_room22_termination_negative.npy",                       
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
                        ]

rooms_combinations = [[[1]], # 1 room,
                      [[0,1],[1,2]], # 2 rooms
                      [[1,0,4],[1,2,6]], #3
                      [[1,0,4,3],[1,0,4,10],[1,2,6,7]], #4
                      [[1,0,4,3,9],[1,0,4,10,11],[1,2,6,7,13]], # 5
                      [[1,0,4,3,9,10],[1,0,4,10,11,5],[1,0,4,10,11,19],[1,2,6,7,13,21],[1,2,6,7,13,11],[1,2,6,7,13,14]],
                      [[1,0,4,3,9,10,11],[1,0,4,10,11,5,19],[1,0,4,10,11,5,13],[1,0,4,10,11,21,13],[1,0,4,10,11,19,13],[1,2,6,7,13,21,22],[1,2,6,7,13,11,10],[1,2,6,7,13,14,22]],
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

        experiment = MonteDivDisClassifierExperiment(base_dir=args.base_dir, seed=args.seed)


        experiment.add_test_files(positive_test_files,
                                  negative_test_files,
                                  uncertain_test_files)
        experiment.read_directory(img_dir)

        experiment.room_by_room_train_labelled('climb_down_ladder', 'termination', rooms_combinations)
        

        #input('Experiment done, press any key to exit.')
        
        #num_batch = 1
        #view_acc = experiment.view_false_predictions(positive_test_files, negative_test_files, num_batch)
        #print(f"Viewing {num_batch} of Predictions:")
        #print(f"Accuracy: {view_acc[0]}")
        #print(f"Weighted Accuracy: {view_acc[1]}")
 