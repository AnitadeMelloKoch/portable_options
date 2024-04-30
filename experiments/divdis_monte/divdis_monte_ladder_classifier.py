import multiprocessing
from experiments.divdis_monte.core.monte_divdis_classifier_experiment import MonteDivDisClassifierExperiment
import argparse 
from portable.utils.utils import load_gin_configs
import torch 
import random 

img_dir = "resources/monte_images/"
positive_train_files = [img_dir+"screen_climb_down_ladder_termination_positive.npy"]
negative_train_files = [img_dir+"screen_climb_down_ladder_termination_negative.npy",
                        img_dir+"screen_death_1.npy"]
unlabeled_train_files = [img_dir+"screen_climb_down_ladder_initiation_positive.npy",
                          img_dir+"climb_down_ladder_room0_screen_termination_positive.npy",
                          img_dir+"climb_down_ladder_room0_screen_termination_negative.npy",
                          img_dir+"climb_down_ladder_room2_screen_termination_negative.npy",
                          img_dir+"climb_down_ladder_room3_screen_termination_negative.npy",
                          img_dir+"climb_down_ladder_room4_screen_termination_positive.npy",
                          img_dir+"climb_down_ladder_room4_screen_termination_negative.npy",
                          img_dir+"climb_down_ladder_room6_screen_termination_positive.npy",
                          img_dir+"climb_down_ladder_room6_screen_termination_negative.npy",
                          img_dir+"climb_down_ladder_room7_screen_termination_negative.npy",
                          img_dir+"screen_death_2.npy",
                          img_dir+"screen_death_3.npy",
                          
                          ]

positive_test_files = [
                        img_dir+"climb_down_ladder_room9_screen_termination_positive.npy",
                        img_dir+"climb_down_ladder_room10_screen_termination_positive.npy",
                        img_dir+"climb_down_ladder_room11_screen_termination_positive.npy",
                        img_dir+"climb_down_ladder_room13_screen_termination_positive.npy",
                        img_dir+"climb_down_ladder_room19_screen_termination_positive.npy",
                        img_dir+"climb_down_ladder_room21_screen_termination_positive.npy",
                        img_dir+"climb_down_ladder_room22_screen_termination_positive.npy",
                       ]
negative_test_files = [
                        img_dir+"climb_down_ladder_room8_screen_termination_negative.npy",
                        img_dir+"climb_down_ladder_room9_screen_termination_negative.npy",
                        img_dir+"climb_down_ladder_room10_screen_termination_negative.npy",
                        img_dir+"climb_down_ladder_room11_screen_termination_negative.npy",
                        img_dir+"climb_down_ladder_room13_screen_termination_negative.npy",
                        img_dir+"climb_down_ladder_room14_screen_termination_negative.npy",
                        img_dir+"climb_down_ladder_room19_screen_termination_negative.npy",
                        img_dir+"climb_down_ladder_room21_screen_termination_negative.npy",
                        img_dir+"climb_down_ladder_room22_screen_termination_negative.npy",
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
                                 unlabeled_train_files)

        experiment.train_classifier()
        
        accuracy = experiment.test_classifier(positive_test_files,
                                              negative_test_files,
                                              save=True)
        
        print(f"Accuracy: {accuracy[0]}")
        print(f"Weighted Accuracy: {accuracy[1]}")
 