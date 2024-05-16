import logging
import multiprocessing

import numpy as np
from experiments.divdis_monte.core.monte_divdis_classifier_experiment import MonteDivDisClassifierExperiment
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
                            img_dir+"climb_down_ladder_room0_initiation_positive.npy",
                         img_dir+"climb_down_ladder_room0_initiation_negative.npy",
        ]
room_list = [0,4,3,9,8,10,11,5,  13,7,6,14,22,21,19,18]

unlabelled_train_files = [
                        #0
                        [
                         #img_dir+"climb_down_ladder_room0_initiation_positive.npy",
                         #img_dir+"climb_down_ladder_room0_initiation_negative.npy",
                         img_dir+"climb_down_ladder_room0_termination_negative.npy",
                         img_dir+"climb_down_ladder_room0_uncertain.npy"],
                         #4
                         [img_dir+"climb_down_ladder_room4_initiation_negative.npy",
                         img_dir+"climb_down_ladder_room4_termination_negative.npy",
                         img_dir+"climb_down_ladder_room4_uncertain.npy"],
                         #3
                         [img_dir+"climb_down_ladder_room3_initiation_positive.npy",
                          img_dir+"climb_down_ladder_room3_initiation_negative.npy",
                          img_dir+"climb_down_ladder_room3_termination_negative.npy",
                          img_dir+"climb_down_ladder_room3_uncertain.npy"],
                         #9
                         [
                          img_dir+"climb_down_ladder_room9_initiation_negative.npy",
                          img_dir+"climb_down_ladder_room9_termination_positive.npy",
                          img_dir+"climb_down_ladder_room9_termination_negative.npy",
                          img_dir+"climb_down_ladder_room9_uncertain.npy"],
                         #8
                         [img_dir+"room8_walk_around.npy"],
                         #10
                         [img_dir+"climb_down_ladder_room10_initiation_negative.npy",
                         img_dir+"climb_down_ladder_room10_termination_negative.npy",
                         img_dir+"climb_down_ladder_room10_uncertain.npy"],
                         #11
                         [
                          img_dir+"climb_down_ladder_room11_initiation_negative.npy",
                          img_dir+"climb_down_ladder_room11_termination_negative.npy",
                          img_dir+"climb_down_ladder_room11_uncertain.npy"],
                         #5
                         [img_dir+"climb_down_ladder_room5_initiation_positive.npy",
                         img_dir+"climb_down_ladder_room5_initiation_negative.npy",
                         img_dir+"climb_down_ladder_room5_termination_negative.npy",
                         img_dir+"climb_down_ladder_room5_uncertain.npy"],
                         #12
                         #13
                         [
                          img_dir+"climb_down_ladder_room13_initiation_negative.npy",
                          img_dir+"climb_down_ladder_room13_termination_negative.npy",
                          img_dir+"climb_down_ladder_room13_uncertain.npy"],
                         #7
                         [
                          img_dir+"climb_down_ladder_room7_initiation_negative.npy",
                          img_dir+"climb_down_ladder_room7_initiation_positive.npy",
                          img_dir+"climb_down_ladder_room7_termination_negative.npy",
                          img_dir+"climb_down_ladder_room7_uncertain.npy"],
                         #6
                         [img_dir+"climb_down_ladder_room6_initiation_negative.npy",
                         img_dir+"climb_down_ladder_room6_termination_positive.npy",
                         img_dir+"climb_down_ladder_room6_termination_negative.npy",
                         img_dir+"climb_down_ladder_room6_uncertain.npy"],
                         #14
                         [img_dir+"climb_down_ladder_room14_initiation_negative.npy",
                         img_dir+"climb_down_ladder_room14_initiation_positive.npy",
                         img_dir+"climb_down_ladder_room14_termination_negative.npy",
                         img_dir+"climb_down_ladder_room14_uncertain.npy"],
                         #22
                         [img_dir+"climb_down_ladder_room22_initiation_negative.npy",
                         img_dir+"climb_down_ladder_room22_termination_negative.npy",
                         img_dir+"climb_down_ladder_room22_uncertain.npy"],
                         #21
                         [img_dir+"climb_down_ladder_room21_initiation_negative.npy",
                         img_dir+"climb_down_ladder_room21_termination_positive.npy",
                         img_dir+"climb_down_ladder_room21_termination_negative.npy",
                         img_dir+"climb_down_ladder_room21_uncertain.npy"],
                         #19
                         [img_dir+"climb_down_ladder_room19_initiation_negative.npy",
                         img_dir+"climb_down_ladder_room19_termination_positive.npy",
                         img_dir+"climb_down_ladder_room19_termination_negative.npy",
                         img_dir+"climb_down_ladder_room19_uncertain.npy"],
                        #18
                         [img_dir+"room18_walk_around.npy",]
                          ]

positive_test_files = [img_dir+"climb_down_ladder_room6_termination_positive.npy",
                       img_dir+"climb_down_ladder_room9_termination_positive.npy",
                       img_dir+"climb_down_ladder_room19_termination_positive.npy",
                       img_dir+"climb_down_ladder_room21_termination_positive.npy",
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
        
        experiment = MonteDivDisClassifierExperiment(base_dir=args.base_dir,
                                                        seed=args.seed)

        experiment.add_datafiles(positive_train_files,
                                 negative_train_files,
                                 initial_unlabelled_train_files)
        experiment.classifier.train(300)

        print("Training on room 1 only")
        logging.info("Training on room 1 only")
        accuracy = experiment.test_classifier(positive_test_files,
                                              negative_test_files)
        uncertainty = experiment.test_uncertainty(uncertain_test_files)
        print(f"Accuracy: {accuracy[0]}")
        print(f"Weighted Accuracy: {accuracy[1]}")
        print(f"Uncertainty: {uncertainty}")

        for room_idx in range(len(room_list)):
            room = room_list[room_idx]
            print(f"Training on room {room}")
            logging.info(f"Training on room {room}")
            cur_room_unlab = unlabelled_train_files[room_idx]
            cur_room_unlab = [np.load(file) for file in cur_room_unlab]
            cur_room_unlab = [img for list in cur_room_unlab for img in list]
            cur_room_unlab = [torch.from_numpy(img).float().squeeze() for img in cur_room_unlab]
            experiment.classifier.dataset.add_unlabelled_data(cur_room_unlab)
            experiment.classifier.train(30)
                
            accuracy = experiment.test_classifier(positive_test_files, negative_test_files)
            uncertainty = experiment.test_uncertainty(uncertain_test_files)
                                                        
            print(f"Accuracy: {accuracy[0]}")
            print(f"Weighted Accuracy: {accuracy[1]}")
            print(f"Uncertainty: {uncertainty}")

        #num_batch = 1
        #view_acc = experiment.view_false_predictions(positive_test_files, negative_test_files, num_batch)
        #print(f"Viewing {num_batch} of Predictions:")
        #print(f"Accuracy: {view_acc[0]}")
        #print(f"Weighted Accuracy: {view_acc[1]}")
 