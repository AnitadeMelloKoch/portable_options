import argparse
import os
import random
import re
import warnings
import torch

#from evaluators import DivDisEvaluatorClassifier
from evaluation.evaluators.divdis_evaluator_classifier import DivDisEvaluatorClassifier
#from experiments.divdis_minigrid.core.advanced_minigrid_factored_divdis_classifier_experiment import \
#    AdvancedMinigridFactoredDivDisClassifierExperiment
from portable.option.divdis.divdis_classifier import DivDisClassifier
from portable.utils.utils import load_gin_configs, set_seed


img_dir = "resources/monte_images/"


def get_sorted_filenames(directory):
    filenames = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            filenames.append(file)
    filenames.sort()
    
    return filenames

data_filenames = get_sorted_filenames(img_dir)


positive_train_files = [img_dir+"climb_down_ladder_room1_termination_positive.npy",
                        img_dir+"climb_down_ladder_room6_termination_positive.npy",
                        img_dir+"climb_down_ladder_room6_1_termination_positive.npy",
                        ]

negative_train_files = [img_dir+"climb_down_ladder_room1_termination_negative.npy",
                        img_dir+"climb_down_ladder_room1_1_termination_negative.npy",
                        img_dir+"screen_death_1.npy",
                        img_dir+"screen_death_2.npy",
                        img_dir+"screen_death_3.npy",
                        img_dir+"screen_death_4.npy",
                        img_dir+"climb_down_ladder_room2_termination_negative.npy",
                        img_dir+"climb_down_ladder_room6_termination_negative.npy",
                        img_dir+"climb_down_ladder_room2_1_termination_negative.npy",
                        img_dir+"climb_down_ladder_room6_1_termination_negative.npy",
                        ]
unlabelled_train_files = [
    # 0
    img_dir + "climb_up_ladder_room0_termination_positive.npy",
     img_dir + "climb_up_ladder_room0_termination_negative.npy",
     img_dir + "climb_up_ladder_room0_uncertain.npy",
     img_dir+"climb_down_ladder_room0_termination_negative.npy",
     img_dir+"climb_down_ladder_room0_uncertain.npy",
    # 4
    img_dir + "climb_up_ladder_room4_termination_negative.npy",
     img_dir + "climb_up_ladder_room4_uncertain.npy",
     img_dir+"climb_down_ladder_room4_termination_negative.npy",
     img_dir+"climb_down_ladder_room4_uncertain.npy",
     img_dir + "move_right_enemy_room4_termination_positive.npy",
     img_dir + "move_right_enemy_room4_termination_negative.npy",
     img_dir + "move_left_enemy_room4_termination_positive.npy",
     img_dir + "move_left_enemy_room4_termination_negative.npy",
    # 3
    img_dir + "climb_up_ladder_room3_termination_positive.npy",
     img_dir + "climb_up_ladder_room3_termination_negative.npy",
     img_dir + "climb_up_ladder_room3_uncertain.npy",
     img_dir+"climb_down_ladder_room3_termination_negative.npy",
     img_dir+"climb_down_ladder_room3_uncertain.npy",
     img_dir + "move_right_enemy_room3_termination_positive.npy",
     img_dir + "move_right_enemy_room3_termination_negative.npy",
     img_dir + "move_left_enemy_room3_termination_positive.npy",
     img_dir + "move_left_enemy_room3_termination_negative.npy",
    # 9
    img_dir + "climb_up_ladder_room9_termination_negative.npy",
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
     img_dir + "move_left_enemy_room9right_termination_negative.npy",
    # 8
    img_dir + "room8_walk_around.npy",
    # 10
    img_dir + "climb_up_ladder_room10_termination_negative.npy",
     img_dir + "climb_up_ladder_room10_uncertain.npy",
     img_dir+"climb_down_ladder_room10_termination_negative.npy",
     img_dir+"climb_down_ladder_room10_termination_positive.npy",
     img_dir+"climb_down_ladder_room10_uncertain.npy",
    # 11
    img_dir + "climb_up_ladder_room11_termination_negative.npy",
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
     img_dir + "move_left_enemy_room11right_termination_negative.npy",
    # 5
    img_dir + "climb_up_ladder_room5_termination_positive.npy",
     #img_dir + "climb_up_ladder_room5_termination_negative.npy",
     img_dir + "climb_up_ladder_room5_uncertain.npy",
     img_dir+"climb_down_ladder_room5_termination_negative.npy",
     img_dir+"climb_down_ladder_room5_uncertain.npy",
     img_dir + "move_right_enemy_room5_termination_positive.npy",
     img_dir + "move_right_enemy_room5_termination_negative.npy",
     img_dir + "move_left_enemy_room5_termination_positive.npy",
     img_dir + "move_left_enemy_room5_termination_negative.npy",
    # 13
    img_dir + "climb_up_ladder_room13_termination_negative.npy",
     img_dir + "climb_up_ladder_room13_uncertain.npy",
     img_dir+"climb_down_ladder_room13_termination_negative.npy",
     img_dir+"climb_down_ladder_room13_uncertain.npy",
     img_dir + "move_right_enemy_room13_termination_positive.npy",
     img_dir + "move_right_enemy_room13_termination_negative.npy",
     img_dir + "move_left_enemy_room13_termination_positive.npy",
     img_dir + "move_left_enemy_room13_termination_negative.npy",
    # 7
    img_dir + "climb_up_ladder_room7_termination_positive.npy",
     img_dir + "climb_up_ladder_room7_termination_negative.npy",
     img_dir + "climb_up_ladder_room7_uncertain.npy",
     img_dir+"climb_down_ladder_room7_termination_negative.npy",
     img_dir+"climb_down_ladder_room7_uncertain.npy",
    # 6
    img_dir + "climb_up_ladder_room6_termination_negative.npy",
     img_dir + "climb_up_ladder_room6_uncertain.npy",
     img_dir+"climb_down_ladder_room6_termination_positive.npy",
     img_dir+"climb_down_ladder_room6_termination_negative.npy",
     img_dir+"climb_down_ladder_room6_uncertain.npy",
    # 2
    img_dir + "climb_up_ladder_room2_termination_positive.npy",
     img_dir + "climb_up_ladder_room2_termination_negative.npy",
     img_dir + "climb_up_ladder_room2_uncertain.npy",
     img_dir+"climb_down_ladder_room2_termination_negative.npy",
     img_dir+"climb_down_ladder_room2_uncertain.npy",
     img_dir + "move_right_enemy_room2_termination_positive.npy",
     img_dir + "move_right_enemy_room2_termination_negative.npy",
     img_dir + "move_left_enemy_room2_termination_positive.npy",
     img_dir + "move_left_enemy_room2_termination_negative.npy",
    # 14
    img_dir + "climb_up_ladder_room14_termination_positive.npy",
     img_dir + "climb_up_ladder_room14_termination_negative.npy",
     img_dir + "climb_up_ladder_room14_uncertain.npy",
     img_dir+"climb_down_ladder_room14_termination_negative.npy",
     img_dir+"climb_down_ladder_room14_uncertain.npy",
    # 22
    img_dir + "climb_up_ladder_room22_termination_negative.npy",
     img_dir + "climb_up_ladder_room22_uncertain.npy",
     img_dir+"climb_down_ladder_room22_termination_negative.npy",
     img_dir+"climb_down_ladder_room22_termination_positive.npy",
     img_dir+"climb_down_ladder_room22_uncertain.npy",
     img_dir + "move_right_enemy_room22_termination_positive.npy",
     img_dir + "move_right_enemy_room22_termination_negative.npy",
     img_dir + "move_left_enemy_room22_termination_positive.npy",
     img_dir + "move_left_enemy_room22_termination_negative.npy",
    # 21
    img_dir + "climb_up_ladder_room21_termination_negative.npy",
     img_dir + "climb_up_ladder_room21_uncertain.npy",
     img_dir+"climb_down_ladder_room21_termination_positive.npy",
     img_dir+"climb_down_ladder_room21_termination_negative.npy",
     img_dir+"climb_down_ladder_room21_uncertain.npy",
     img_dir + "move_right_enemy_room21_termination_positive.npy",
     img_dir + "move_right_enemy_room21_termination_negative.npy",
     img_dir + "move_left_enemy_room21_termination_positive.npy",
     img_dir + "move_left_enemy_room21_termination_negative.npy",
    # 19
    img_dir + "climb_up_ladder_room19_termination_negative.npy",
     img_dir + "climb_up_ladder_room19_uncertain.npy",
     img_dir+"climb_down_ladder_room19_termination_positive.npy",
     img_dir+"climb_down_ladder_room19_termination_negative.npy",
     img_dir+"climb_down_ladder_room19_uncertain.npy",
    # 18
    img_dir + "room18_walk_around.npy",
     img_dir + "move_left_enemy_room18_termination_positive.npy",
     img_dir + "move_left_enemy_room18_termination_negative.npy"
]
unlabelled_train_files = []
unlabelled_train_files = unlabelled_train_files = [img_dir+file for file in data_filenames if (file not in positive_train_files) and (file not in negative_train_files)]
sample_rate = 1
unlabelled_train_files = random.sample(unlabelled_train_files, int(sample_rate*len(unlabelled_train_files)))


positive_test_files = [img_dir+"climb_down_ladder_room1_termination_positive.npy",
                       img_dir+"climb_down_ladder_room6_termination_positive.npy",
                       img_dir+"climb_down_ladder_room9_termination_positive.npy",
                       img_dir+"climb_down_ladder_room10_termination_positive.npy",
                       img_dir+"climb_down_ladder_room19_termination_positive.npy",
                       img_dir+"climb_down_ladder_room21_termination_positive.npy",
                       img_dir+"climb_down_ladder_room22_termination_positive.npy",
                       
                       img_dir+"climb_down_ladder_room6_1_termination_positive.npy",
                       img_dir+"climb_down_ladder_room9_1_termination_positive.npy",
                       img_dir+"climb_down_ladder_room10_1_termination_positive.npy",
                       img_dir+"climb_down_ladder_room19_1_termination_positive.npy",
                       img_dir+"climb_down_ladder_room21_1_termination_positive.npy",
                       img_dir+"climb_down_ladder_room22_1_termination_positive.npy",
                       ]
negative_test_files = [img_dir+"screen_death_1.npy",
                       img_dir+"screen_death_2.npy",
                       img_dir+"screen_death_3.npy",
                       img_dir+"screen_death_4.npy",
                       img_dir+"climb_down_ladder_room1_termination_negative.npy",
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
                       
                       img_dir+"climb_down_ladder_room1_1_termination_negative.npy",
                       img_dir+"climb_down_ladder_room2_1_termination_negative.npy",
                       img_dir+"climb_down_ladder_room3_1_termination_negative.npy",
                       img_dir+"climb_down_ladder_room4_1_termination_negative.npy",
                       img_dir+"climb_down_ladder_room5_1_termination_negative.npy",
                       img_dir+"climb_down_ladder_room6_1_termination_negative.npy",
                       img_dir+"climb_down_ladder_room7_1_termination_negative.npy",
                       img_dir+"climb_down_ladder_room9_1_termination_negative.npy",
                       img_dir+"climb_down_ladder_room10_1_termination_negative.npy",
                       img_dir+"climb_down_ladder_room11_1_termination_negative.npy",
                       img_dir+"climb_down_ladder_room13_1_termination_negative.npy",
                       img_dir+"climb_down_ladder_room14_1_termination_negative.npy",
                       img_dir+"climb_down_ladder_room19_1_termination_negative.npy",
                       img_dir+"climb_down_ladder_room21_1_termination_negative.npy",
                       img_dir+"climb_down_ladder_room22_1_termination_negative.npy",        
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

#nlabelled_train_files = positive_test_files + negative_test_files + uncertain_test_files
#nlabelled_train_files = [file for file in unlabelled_train_files if (file not in positive_train_files) and (file not in negative_train_files)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--classifier_dir", type=str, required=True)
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config_file", nargs='+', type=str, required=True)
    parser.add_argument("--gin_bindings", default=[], help='Gin bindings to override the values' + 
                ' set in the config files (e.g. "DQNAgent.epsilon_train=0.1",' +
                ' "create_atari_environment.game_name="Pong"").')
    args = parser.parse_args()
    load_gin_configs(args.config_file, args.gin_bindings)


    warnings.filterwarnings("ignore", message="Input Tensor 0 did not already require gradients")
    warnings.filterwarnings("ignore", message="Setting forward, backward hooks and attributes")
    warnings.filterwarnings("ignore", message="Lazy modules are a new feature under heavy development")


    set_seed(args.seed)

    base_dir = os.path.join(args.base_dir, str(args.seed))

    classifier = DivDisClassifier(log_dir=os.path.join(base_dir, "logs"))
    classifier.add_data(positive_train_files,
                        negative_train_files,
                        unlabelled_train_files)
    classifier.set_class_weights()
    classifier.train(130, progress_bar=True)

    

    evaluator = DivDisEvaluatorClassifier(
                    classifier,
                    base_dir=base_dir,
                    test_batch_size=256)
    evaluator.add_test_files(positive_test_files, negative_test_files)
    acc_pos, acc_neg, acc, weighted_acc = evaluator.test_classifier()
    print(f"weighted_acc: {weighted_acc}")
    print(f"raw acc:      {acc}")
    print(f"acc_pos:      {acc_pos}")
    print(f"acc_neg:      {acc_neg}")

    evaluator.evaluate_images(250)

    #evaluator.add_true_from_files(positive_test_files)
    #evaluator.add_false_from_files(negative_test_files)
    #evaluator.evaluate(2)

    # print head complexity
    #print(evaluator.get_head_complexity())