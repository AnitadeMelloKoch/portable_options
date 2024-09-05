import argparse 
import logging 
import random 

import numpy as np 

from experiments.divdis_monte.core.divdis_monte_classifier_experiment import MonteDivDisClassifierExperiment
from portable.utils.utils import load_gin_configs

positive_train_files = ["resources/monte_images/lasers1_toleft_room0_termination_positive.npy"]
negative_train_files = ["resources/monte_images/lasers1_toleft_room0_termination_negative.npy"]
unlabelled_train_files = ["resources/monte_images/lasers2_toleft_room0_termination_positive.npy",
                          "resources/monte_images/lasers2_toleft_room0_termination_negative.npy",
                          "resources/monte_images/climb_down_ladder_room2_initiation_positive.npy",
                          "resources/monte_images/lasers_wait_disappear_room7_termination_positive.npy",
                          "resources/monte_images/lasers_wait_disappear_room7_termination_negative.npy",]

positive_test_files = ["resources/monte_images/lasers1_toleft_room7_termination_positive.npy",
                       "resources/monte_images/lasers2_toleft_room7_termination_positive.npy",
                       "resources/monte_images/lasers_toleft_room12_termination_positive.npy"]
negative_test_files = ["resources/monte_images/lasers1_toleft_room7_termination_negative.npy",
                       "resources/monte_images/lasers2_toleft_room7_termination_negative.npy",
                       "resources/monte_images/lasers_toleft_room12_termination_negative.npy"]

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
    
    experiment = MonteDivDisClassifierExperiment(base_dir=args.base_dir,
                                                 seed=args.seed)
    
    experiment.add_train_files(positive_train_files,
                               negative_train_files,
                               unlabelled_train_files)
    experiment.add_test_files(positive_test_files,
                              negative_test_files)
    
    experiment.train_classifier(150)
    
    accuracy_pos, accuracy_neg, accuracy, weighted_acc = experiment.test_classifier()
    
    print("=========================================================")
    print("positive accuracy", accuracy_pos)
    print("negative accuracy", accuracy_neg)
    print("raw accuracy", accuracy)
    print("weighted accuracy", weighted_acc)
    print("=========================================================")
    
    


