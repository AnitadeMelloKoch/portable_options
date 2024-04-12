from experiments.divdis_monte.core.monte_divdis_classifier_experiment import MonteDivDisClassifierExperiment
import argparse 
from portable.utils.utils import load_gin_configs
import torch 
import random 

img_dir = "resources/monte_images/"
positive_train_files = [img_dir+"screen_climb_down_ladder_termination_positive.npy"]
negative_train_files = [img_dir+"screen_climb_down_ladder_termination_negative.npy",
                        img_dir+"screen_death_1.npy"]
unlabelled_train_files = [img_dir+"screen_climb_down_ladder_initiation_positive.npy",
                          img_dir+"screen_death_2.npy",
                          img_dir+"screen_death_3.npy",
                        #  img_dir+"room2_agent_bottom_ladder.npy",
                          img_dir+"room4_move_left_spider_initiation_positive.npy",
                          img_dir+"room4_move_left_spider_initiation_positive.npy",
                          img_dir+"climb_down_ladder_room0_screen_termination_positive.npy",
                          img_dir+"climb_down_ladder_room4_screen_termination_positive.npy",
                          img_dir+"climb_down_ladder_room0_screen_termination_negative.npy",
                          img_dir+"climb_down_ladder_room4_screen_termination_negative.npy",
                          img_dir+"screen_death_4.npy"]

positive_test_files = [img_dir+"climb_down_ladder_room0_screen_termination_positive.npy",
                       img_dir+"climb_down_ladder_room4_screen_termination_positive.npy",]
negative_test_files = [img_dir+"climb_down_ladder_room0_screen_termination_negative.npy",
                       img_dir+"climb_down_ladder_room4_screen_termination_negative.npy",
                       img_dir+"screen_death_4.npy",]

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

        experiment.add_datafiles(positive_train_files,
                                 negative_train_files,
                                 unlabelled_train_files)

        experiment.train_classifier()
        
        accuracy = experiment.test_classifier(positive_test_files,
                                              negative_test_files)
        
        print(accuracy)
 