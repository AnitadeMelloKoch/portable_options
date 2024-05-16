import multiprocessing
from experiments.divdis_monte.core.monte_divdis_classifier_experiment import MonteDivDisClassifierExperiment
import argparse 
from portable.utils.utils import load_gin_configs
import torch 
import random 

img_dir = "resources/monte_images/"
# train using room 1 only
positive_train_files = [img_dir+"screen_climb_down_ladder_termination_positive.npy"]
negative_train_files = [img_dir+"screen_climb_down_ladder_termination_negative.npy",
                        #img_dir+"screen_death_1.npy",
                        #img_dir+"screen_death_2.npy",
                        #img_dir+"screen_death_3.npy",
                        #img_dir+"screen_death_4.npy"
                        ]

# unlabeled using room 1, 0, 4, 5, 6, 10 + variety of iamges
unlabeled_train_files = [
                         #img_dir+"screen_climb_down_ladder_initiation_positive.npy", # unlabelled should not contain data from room 1
                         #img_dir+"screen_climb_down_ladder_initiation_negative.npy",
                         #img_dir+"screen_death_1.npy",
                         #img_dir+"screen_death_2.npy",
                         #img_dir+"screen_death_3.npy",
                         #img_dir+"screen_death_4.npy",

                         img_dir+"room8_walk_around.npy",
                         img_dir+"room18_walk_around.npy",

                         img_dir+"climb_down_ladder_room6_termination_positive.npy",
                         img_dir+"climb_down_ladder_room9_termination_positive.npy",
                         img_dir+"climb_down_ladder_room19_termination_positive.npy",
                         img_dir+"climb_down_ladder_room21_termination_positive.npy",

                         img_dir+"climb_down_ladder_room2_termination_negative.npy",
                       img_dir+"climb_down_ladder_room3_termination_negative.npy",
                       img_dir+"climb_down_ladder_room7_termination_negative.npy",
                       img_dir+"climb_down_ladder_room9_termination_negative.npy",
                       img_dir+"climb_down_ladder_room11_termination_negative.npy",
                       img_dir+"climb_down_ladder_room13_termination_negative.npy",
                       img_dir+"climb_down_ladder_room14_termination_negative.npy",
                       img_dir+"climb_down_ladder_room19_termination_negative.npy",
                       img_dir+"climb_down_ladder_room21_termination_negative.npy",
                       img_dir+"climb_down_ladder_room22_termination_negative.npy",     

                       img_dir+"climb_down_ladder_room0_termination_negative.npy",
                       img_dir+"climb_down_ladder_room4_termination_negative.npy",
                       img_dir+"climb_down_ladder_room5_termination_negative.npy",
                       img_dir+"climb_down_ladder_room6_termination_negative.npy",
                       img_dir+"climb_down_ladder_room10_termination_negative.npy",


                        img_dir+"climb_down_ladder_room2_uncertain.npy",
                        img_dir+"climb_down_ladder_room3_uncertain.npy",
                        img_dir+"climb_down_ladder_room7_uncertain.npy",
                        img_dir+"climb_down_ladder_room10_uncertain.npy",
                        img_dir+"climb_down_ladder_room11_uncertain.npy",
                        img_dir+"climb_down_ladder_room13_uncertain.npy",
                        img_dir+"climb_down_ladder_room14_uncertain.npy",

                        img_dir+"climb_down_ladder_room0_uncertain.npy",
                        img_dir+"climb_down_ladder_room4_uncertain.npy",
                        img_dir+"climb_down_ladder_room5_uncertain.npy",
                        img_dir+"climb_down_ladder_room10_uncertain.npy",
                          ]

positive_test_files = [img_dir+"climb_down_ladder_room6_termination_positive.npy",
                       img_dir+"climb_down_ladder_room9_termination_positive.npy",
                       img_dir+"climb_down_ladder_room19_termination_positive.npy",
                       img_dir+"climb_down_ladder_room21_termination_positive.npy",
                       ]
negative_test_files = [img_dir+"climb_down_ladder_room2_termination_negative.npy",
                       img_dir+"climb_down_ladder_room3_termination_negative.npy",
                       img_dir+"climb_down_ladder_room7_termination_negative.npy",
                       img_dir+"climb_down_ladder_room9_termination_negative.npy",
                       img_dir+"climb_down_ladder_room11_termination_negative.npy",
                       img_dir+"climb_down_ladder_room13_termination_negative.npy",
                       img_dir+"climb_down_ladder_room14_termination_negative.npy",
                       img_dir+"climb_down_ladder_room19_termination_negative.npy",
                       img_dir+"climb_down_ladder_room21_termination_negative.npy",
                       img_dir+"climb_down_ladder_room22_termination_negative.npy",     

                       img_dir+"climb_down_ladder_room0_termination_negative.npy",
                       img_dir+"climb_down_ladder_room4_termination_negative.npy",
                       img_dir+"climb_down_ladder_room5_termination_negative.npy",
                       img_dir+"climb_down_ladder_room6_termination_negative.npy",
                       img_dir+"climb_down_ladder_room10_termination_negative.npy",
                        
                       ]
uncertain_test_files = [img_dir+"climb_down_ladder_room2_uncertain.npy",
                        img_dir+"climb_down_ladder_room3_uncertain.npy",
                        img_dir+"climb_down_ladder_room7_uncertain.npy",
                        img_dir+"climb_down_ladder_room10_uncertain.npy",
                        img_dir+"climb_down_ladder_room11_uncertain.npy",
                        img_dir+"climb_down_ladder_room13_uncertain.npy",
                        img_dir+"climb_down_ladder_room14_uncertain.npy",

                        img_dir+"climb_down_ladder_room0_uncertain.npy",
                        img_dir+"climb_down_ladder_room4_uncertain.npy",
                        img_dir+"climb_down_ladder_room5_uncertain.npy",
                        img_dir+"climb_down_ladder_room10_uncertain.npy",
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
        import warnings
        warnings.filterwarnings("ignore", message=".*MessageStream.*")
        
        experiment = MonteDivDisClassifierExperiment(base_dir=args.base_dir,
                                                        seed=args.seed)

        experiment.add_datafiles(positive_train_files,
                                 negative_train_files,
                                 unlabeled_train_files)

        experiment.train_classifier()
        
        accuracy = experiment.test_classifier(positive_test_files,
                                              negative_test_files)
        uncertainty = experiment.test_uncertainty(uncertain_test_files)
                                                 
        
        print(f"Accuracy: {accuracy[0]}")
        print(f"Weighted Accuracy: {accuracy[1]}")
        print(f"Uncertainty: {uncertainty}")

        num_batch = 1
        view_acc = experiment.view_false_predictions(positive_test_files,
                                                     negative_test_files,
                                                     num_batch)
        print(f"Viewing {num_batch} of Predictions:")
        print(f"Accuracy: {view_acc[0]}")
        print(f"Weighted Accuracy: {view_acc[1]}")
 