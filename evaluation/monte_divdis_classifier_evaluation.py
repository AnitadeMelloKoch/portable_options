import argparse
import os
import random
import torch

#from evaluators import DivDisEvaluatorClassifier
from evaluation.evaluators.divdis_evaluator_classifier import DivDisEvaluatorClassifier
#from experiments.divdis_minigrid.core.advanced_minigrid_factored_divdis_classifier_experiment import \
#    AdvancedMinigridFactoredDivDisClassifierExperiment
from portable.option.divdis.divdis_classifier import DivDisClassifier
from portable.utils.utils import load_gin_configs


img_dir = "resources/monte_images/"
positive_train_files = [img_dir+"screen_climb_down_ladder_termination_positive.npy"]
negative_train_files = [img_dir+"screen_climb_down_ladder_termination_negative.npy",
                        img_dir+"screen_death_1.npy",
                        img_dir+"screen_death_2.npy",
                        img_dir+"screen_death_3.npy"]
# unlabeled using room 1, 0, 4, 5, 6, 10 + variety of iamges
unlabeled_train_files = [img_dir+"screen_climb_down_ladder_initiation_positive.npy",
                         img_dir+"screen_climb_down_ladder_initiation_negative.npy",
                         img_dir+"screen_death_4.npy",
                         
                         img_dir+"climb_down_ladder_room0_initiation_positive.npy",
                         img_dir+"climb_down_ladder_room0_initiation_negative.npy",
                         img_dir+"climb_down_ladder_room0_termination_negative.npy",
                         img_dir+"climb_down_ladder_room0_uncertain.npy",
                         
                         img_dir+"climb_down_ladder_room4_initiation_negative.npy",
                         img_dir+"climb_down_ladder_room4_termination_negative.npy",
                         img_dir+"climb_down_ladder_room4_uncertain.npy",

                         img_dir+"climb_down_ladder_room5_initiation_positive.npy",
                         img_dir+"climb_down_ladder_room5_initiation_negative.npy",
                         img_dir+"climb_down_ladder_room5_termination_negative.npy",
                         img_dir+"climb_down_ladder_room5_uncertain.npy",

                         img_dir+"climb_down_ladder_room6_initiation_negative.npy",
                         img_dir+"climb_down_ladder_room6_termination_positive.npy",
                         img_dir+"climb_down_ladder_room6_termination_negative.npy",
                          
                         img_dir+"climb_down_ladder_room10_initiation_negative.npy",
                         img_dir+"climb_down_ladder_room10_termination_negative.npy",
                         img_dir+"climb_down_ladder_room10_uncertain.npy",

                         img_dir+"room8_walk_around.npy",
                         img_dir+"room18_walk_around.npy",
                          ]

positive_test_files = [img_dir+"climb_down_ladder_room9_termination_positive.npy",
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
                       ]
uncertain_test_files = [img_dir+"climb_down_ladder_room2_uncertain.npy",
                        img_dir+"climb_down_ladder_room3_uncertain.npy",
                        img_dir+"climb_down_ladder_room7_uncertain.npy",
                        img_dir+"climb_down_ladder_room10_uncertain.npy",
                        img_dir+"climb_down_ladder_room11_uncertain.npy",
                        img_dir+"climb_down_ladder_room13_uncertain.npy",
                        img_dir+"climb_down_ladder_room14_uncertain.npy",
                        ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--classifier_dir", type=str, required=True)
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--config_file", nargs='+', type=str, required=True)
    parser.add_argument("--gin_bindings", default=[], help='Gin bindings to override the values' + 
                ' set in the config files (e.g. "DQNAgent.epsilon_train=0.1",' +
                ' "create_atari_environment.game_name="Pong"").')
    args = parser.parse_args()
    load_gin_configs(args.config_file, args.gin_bindings)

    classifier = DivDisClassifier(log_dir=args.base_dir+"logs")
    classifier.add_data(positive_train_files,
                        negative_train_files,
                        unlabeled_train_files)
    classifier.train(2000)

    evaluator = DivDisEvaluatorClassifier(
                    classifier,
                    base_dir=args.base_dir)
    evaluator.add_test_files(positive_test_files, negative_test_files)
    evaluator.evaluate_images(25)

    #evaluator.add_true_from_files(positive_test_files)
    #evaluator.add_false_from_files(negative_test_files)
    #evaluator.evaluate(2)

    # print head complexity
    #print(evaluator.get_head_complexity())