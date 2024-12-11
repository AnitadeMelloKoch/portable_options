import logging
import multiprocessing

import numpy as np
from experiments.divdis_monte.core.divdis_monte_classifier_experiment import MonteDivDisClassifierExperiment
import argparse 
from portable.utils.utils import load_gin_configs
import torch 
import random 

img_dir = "resources/monte_images/"
# train using room 1 only
positive_train_files = [img_dir+"screen_climb_down_ladder_termination_positive.npy"]
negative_train_files = [img_dir+"screen_climb_down_ladder_termination_negative.npy",
                        img_dir+"screen_death_1.npy",
                        img_dir+"screen_death_2.npy",
                        img_dir+"screen_death_3.npy",
                        img_dir+"screen_death_4.npy"
                        ]
initial_unlabelled_train_files = [
                            #img_dir+"screen_climb_down_ladder_initiation_positive.npy",
                            #img_dir+"screen_climb_down_ladder_initiation_negative.npy",
                            #img_dir+"climb_down_ladder_room0_initiation_positive.npy",
                            #img_dir+"climb_down_ladder_room0_initiation_negative.npy",
        ]
room_list = [0, 4, 3, 9, 8, 10, 11, 5, #12 here, has nothing
             13, 7, 6, 2, 14, 22, # 23 
             21, 19, 18]

unlabelled_train_files = [
    # 0
    [img_dir + "climb_up_ladder_room0_termination_positive.npy",
     img_dir + "climb_up_ladder_room0_termination_negative.npy",
     img_dir + "climb_up_ladder_room0_uncertain.npy",
     img_dir+"climb_down_ladder_room0_termination_negative.npy",
     img_dir+"climb_down_ladder_room0_uncertain.npy"],
    # 4
    [img_dir + "climb_up_ladder_room4_termination_negative.npy",
     img_dir + "climb_up_ladder_room4_uncertain.npy",
     img_dir+"climb_down_ladder_room4_termination_negative.npy",
     img_dir+"climb_down_ladder_room4_uncertain.npy",
     img_dir + "move_right_enemy_room4_termination_positive.npy",
     img_dir + "move_right_enemy_room4_termination_negative.npy",
     img_dir + "move_left_enemy_room4_termination_positive.npy",
     img_dir + "move_left_enemy_room4_termination_negative.npy"],
    # 3
    [img_dir + "climb_up_ladder_room3_termination_positive.npy",
     img_dir + "climb_up_ladder_room3_termination_negative.npy",
     img_dir + "climb_up_ladder_room3_uncertain.npy",
     img_dir+"climb_down_ladder_room3_termination_negative.npy",
     img_dir+"climb_down_ladder_room3_uncertain.npy",
     img_dir + "move_right_enemy_room3_termination_positive.npy",
     img_dir + "move_right_enemy_room3_termination_negative.npy",
     img_dir + "move_left_enemy_room3_termination_positive.npy",
     img_dir + "move_left_enemy_room3_termination_negative.npy"],
    # 9
    [img_dir + "climb_up_ladder_room9_termination_negative.npy",
     img_dir + "climb_up_ladder_room9_uncertain.npy",
     img_dir+"climb_down_ladder_room9_termination_positive.npy",
     img_dir+"climb_down_ladder_room9_termination_negative.npy",
     img_dir+"climb_down_ladder_room9_uncertain.npy",
     img_dir + "move_right_enemy_room9left_termination_positive.npy",
     img_dir + "move_right_enemy_room9left_termination_negative.npy",
     img_dir + "move_right_enemy_room9right_termination_positive.npy",
     img_dir + "move_right_enemy_room9right_termination_negative.npy",
     img_dir + "move_left_enemy_room9left_termination_positive.npy",
     img_dir + "move_left_enemy_room9left_termination_negative.npy",
     img_dir + "move_left_enemy_room9right_termination_positive.npy",
     img_dir + "move_left_enemy_room9right_termination_negative.npy"],
    # 8
    [img_dir + "room8_walk_around.npy"],
    # 10
    [img_dir + "climb_up_ladder_room10_termination_negative.npy",
     img_dir + "climb_up_ladder_room10_uncertain.npy",
     img_dir+"climb_down_ladder_room10_termination_negative.npy",
     img_dir+"climb_down_ladder_room10_termination_positive.npy",
     img_dir+"climb_down_ladder_room10_uncertain.npy"],
    # 11
    [img_dir + "climb_up_ladder_room11_termination_negative.npy",
     img_dir + "climb_up_ladder_room11_uncertain.npy",
     img_dir+"climb_down_ladder_room11_termination_negative.npy",
     img_dir+"climb_down_ladder_room11_uncertain.npy",
     img_dir + "move_right_enemy_room11left_termination_positive.npy",
     img_dir + "move_right_enemy_room11left_termination_negative.npy",
     img_dir + "move_right_enemy_room11right_termination_positive.npy",
     img_dir + "move_right_enemy_room11right_termination_negative.npy",
     img_dir + "move_left_enemy_room11left_termination_positive.npy",
     img_dir + "move_left_enemy_room11left_termination_negative.npy",
     img_dir + "move_left_enemy_room11right_termination_positive.npy",
     img_dir + "move_left_enemy_room11right_termination_negative.npy"],
    # 5
    [img_dir + "climb_up_ladder_room5_termination_positive.npy",
     img_dir + "climb_up_ladder_room5_termination_negative.npy",
     img_dir + "climb_up_ladder_room5_uncertain.npy",
     img_dir+"climb_down_ladder_room5_termination_negative.npy",
     img_dir+"climb_down_ladder_room5_uncertain.npy",
     img_dir + "move_right_enemy_room5_termination_positive.npy",
     img_dir + "move_right_enemy_room5_termination_negative.npy",
     img_dir + "move_left_enemy_room5_termination_positive.npy",
     img_dir + "move_left_enemy_room5_termination_negative.npy"],
    # 13
    [img_dir + "climb_up_ladder_room13_termination_negative.npy",
     img_dir + "climb_up_ladder_room13_uncertain.npy",
     img_dir+"climb_down_ladder_room13_termination_negative.npy",
     img_dir+"climb_down_ladder_room13_uncertain.npy",
     img_dir + "move_right_enemy_room13_termination_positive.npy",
     img_dir + "move_right_enemy_room13_termination_negative.npy",
     img_dir + "move_left_enemy_room13_termination_positive.npy",
     img_dir + "move_left_enemy_room13_termination_negative.npy"],
    # 7
    [img_dir + "climb_up_ladder_room7_termination_positive.npy",
     img_dir + "climb_up_ladder_room7_termination_negative.npy",
     img_dir + "climb_up_ladder_room7_uncertain.npy",
     img_dir+"climb_down_ladder_room7_termination_negative.npy",
     img_dir+"climb_down_ladder_room7_uncertain.npy"],
    # 6
    [img_dir + "climb_up_ladder_room6_termination_negative.npy",
     img_dir + "climb_up_ladder_room6_uncertain.npy",
     img_dir+"climb_down_ladder_room6_termination_positive.npy",
     img_dir+"climb_down_ladder_room6_termination_negative.npy",
     img_dir+"climb_down_ladder_room6_uncertain.npy"],
    # 2
    [img_dir + "climb_up_ladder_room2_termination_positive.npy",
     img_dir + "climb_up_ladder_room2_termination_negative.npy",
     img_dir + "climb_up_ladder_room2_uncertain.npy",
     img_dir+"climb_down_ladder_room2_termination_negative.npy",
     img_dir+"climb_down_ladder_room2_uncertain.npy",
     img_dir + "move_right_enemy_room2_termination_positive.npy",
     img_dir + "move_right_enemy_room2_termination_negative.npy",
     img_dir + "move_left_enemy_room2_termination_positive.npy",
     img_dir + "move_left_enemy_room2_termination_negative.npy"],
    # 14
    [img_dir + "climb_up_ladder_room14_termination_positive.npy",
     img_dir + "climb_up_ladder_room14_termination_negative.npy",
     img_dir + "climb_up_ladder_room14_uncertain.npy",
     img_dir+"climb_down_ladder_room14_termination_negative.npy",
     img_dir+"climb_down_ladder_room14_uncertain.npy"],
    # 22
    [img_dir + "climb_up_ladder_room22_termination_negative.npy",
     img_dir + "climb_up_ladder_room22_uncertain.npy",
     img_dir+"climb_down_ladder_room22_termination_negative.npy",
     img_dir+"climb_down_ladder_room22_termination_positive.npy",
     img_dir+"climb_down_ladder_room22_uncertain.npy",
     img_dir + "move_right_enemy_room22_termination_positive.npy",
     img_dir + "move_right_enemy_room22_termination_negative.npy",
     img_dir + "move_left_enemy_room22_termination_positive.npy",
     img_dir + "move_left_enemy_room22_termination_negative.npy"],
    # 21
    [img_dir + "climb_up_ladder_room21_termination_negative.npy",
     img_dir + "climb_up_ladder_room21_uncertain.npy",
     img_dir+"climb_down_ladder_room21_termination_positive.npy",
     img_dir+"climb_down_ladder_room21_termination_negative.npy",
     img_dir+"climb_down_ladder_room21_uncertain.npy",
     img_dir + "move_right_enemy_room21_termination_positive.npy",
     img_dir + "move_right_enemy_room21_termination_negative.npy",
     img_dir + "move_left_enemy_room21_termination_positive.npy",
     img_dir + "move_left_enemy_room21_termination_negative.npy"],
    # 19
    [img_dir + "climb_up_ladder_room19_termination_negative.npy",
     img_dir + "climb_up_ladder_room19_uncertain.npy",
     img_dir+"climb_down_ladder_room19_termination_positive.npy",
     img_dir+"climb_down_ladder_room19_termination_negative.npy",
     img_dir+"climb_down_ladder_room19_uncertain.npy"],
    # 18
    [img_dir + "room18_walk_around.npy",
     img_dir + "move_left_enemy_room18_termination_positive.npy",
     img_dir + "move_left_enemy_room18_termination_negative.npy"]
]

positive_test_files = [img_dir+"climb_down_ladder_room6_termination_positive.npy",
                       img_dir+"climb_down_ladder_room9_termination_positive.npy",
                       img_dir+"climb_down_ladder_room10_termination_positive.npy",
                       img_dir+"climb_down_ladder_room19_termination_positive.npy",
                       img_dir+"climb_down_ladder_room21_termination_positive.npy",
                       img_dir+"climb_down_ladder_room22_termination_positive.npy",
                       ]
negative_test_files = [img_dir+"climb_down_ladder_room0_termination_negative.npy",
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

        seeds = [args.seed * i for i in range(1, 6)]
        room_histories = []
        additional_histories = []

        for seed in seeds:
            print(f"Running experiment for seed {seed}")
        
            experiment = MonteDivDisClassifierExperiment(base_dir=args.base_dir,
                                                            seed=seed)

            experiment.add_train_files(positive_train_files,
                                       negative_train_files,
                                       initial_unlabelled_train_files)
            experiment.add_test_files(positive_test_files,
                                      negative_test_files,
                                      uncertain_test_files)
            
            experiment.train_classifier(experiment.initial_epochs)

            print("Training on room 1 only")
            logging.info("Training on room 1 only")
            accuracy_pos, accuracy_neg, accuracy, weighted_acc = experiment.test_classifier()
            uncertainty = experiment.test_uncertainty()
                                                        
            print(f"Weighted Accuracy: {weighted_acc}")
            print(f"Accuracy: {accuracy}")
            print(f"Uncertainty: {uncertainty}")

            best_weighted_acc = np.max(weighted_acc)
            best_head_idx = np.argmax(weighted_acc)
            best_accuracy = accuracy[best_head_idx]
            best_true_acc = accuracy_pos[best_head_idx]
            best_false_acc = accuracy_neg[best_head_idx]
            best_head_uncertainty = uncertainty[best_head_idx]

            history = {
            'weighted_accuracy': [best_weighted_acc],
            'raw_accuracy': [best_accuracy],
            'true_accuracy': [best_true_acc], 
            'false_accuracy': [best_false_acc],
            'uncertainty': [best_head_uncertainty]
        }

            history = experiment.room_by_room_train(room_list, unlabelled_train_files, history)
            room_histories.append(history)

            print("All unlabelled rooms added, now running additional training loops")
            logging.info("All unlabelled rooms added, now running additional training loops")

            history = experiment.additional_train()
            additional_histories.append(history)
    
        experiment.plot_metrics(room_histories, 'room', 'avg_room_train_metrics')
        experiment.plot_metrics(additional_histories, 'additional train loops', 'avg_additional_train_metrics')
        
        #num_batch = 1
        #view_acc = experiment.view_false_predictions(positive_test_files, negative_test_files, num_batch)
        #print(f"Viewing {num_batch} of Predictions:")
        #print(f"Accuracy: {view_acc[0]}")
        #print(f"Weighted Accuracy: {view_acc[1]}")
 