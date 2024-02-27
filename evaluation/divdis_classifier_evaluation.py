import argparse
import os
import random

import multiprocess as mp
import torch

from evaluation.evaluators import DivDisEvaluatorClassifier
from experiments.divdis_minigrid.core.advanced_minigrid_factored_divdis_classifier_experiment import \
    AdvancedMinigridFactoredDivDisClassifierExperiment
from portable.option.divdis.divdis_classifier import DivDisClassifier
from portable.utils.utils import load_gin_configs

'''positive_train_files = ["resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_0_initiation_positive.npy"]
negative_train_files = ["resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_0_initiation_negative.npy"]
unlabelled_train_files = ["resources/factored_minigrid_images/adv_doorkey_8x8_v2_openbluedoor_doorblue_1_initiation_positive.npy",
                          "resources/factored_minigrid_images/adv_doorkey_8x8_v2_openbluedoor_doorblue_0_initiation_negative.npy",
                          "resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_2_initiation_positive.npy",
                          "resources/factored_minigrid_images/adv_doorkey_8x8_v2_openyellowdoor_dooryellow_1_initiation_negative.npy"]

positive_test_files = [
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_3_initiation_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_4_initiation_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_5_initiation_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_6_initiation_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_7_initiation_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_8_initiation_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_9_initiation_positive.npy",
                      ]
negative_test_files = [
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_3_initiation_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_4_initiation_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_5_initiation_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_6_initiation_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_7_initiation_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_8_initiation_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_9_initiation_negative.npy",
                      ]'''

positive_train_files = [
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_0_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_0_1_termination_positive.npy",
    ]
negative_train_files = [
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_0_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_0_1_termination_negative.npy",
    ]
unlabelled_train_files = [
    # "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getbluekey_doorblue_1_1_termination_positive.npy",
    # "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getbluekey_doorblue_1_1_termination_negative.npy",
    # "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getgreenkey_doorgreen_1_1_termination_negative.npy",
    # "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getgreenkey_doorgreen_1_1_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_2_1_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_1_1_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_2_1_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_1_1_termination_negative.npy",
                          ]

positive_test_files = [
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_5_1_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_6_1_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_7_1_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_8_1_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_9_1_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_10_1_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_11_1_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_3_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_4_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_5_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_6_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_7_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_8_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_9_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_10_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_11_termination_positive.npy",
                      ]
negative_test_files = [
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_5_1_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_6_1_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_7_1_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_8_1_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_9_1_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_10_1_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_11_1_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_3_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_4_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_5_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_6_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_7_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_8_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_9_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_10_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_11_termination_negative.npy",
                      ]

def transform(x):
    x = x/torch.tensor([7,7,1,1,5,7,7,5,7,7,5,7,7,5,7,7,5,7,7,5, 7,7,4,7,7,7])
    return x

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

    classifier = DivDisClassifier()
    classifier.add_data(positive_train_files,
                        negative_train_files,
                        unlabelled_train_files)
    classifier.train(50)

    evaluator = DivDisEvaluatorClassifier(
                    classifier,
                    batch_size=1000,
                    base_dir=args.base_dir)

    evaluator.add_test_files(positive_test_files, negative_test_files)
    evaluator.test_dataset.set_transform_function(transform)

    evaluator.evaluate(num_features=26, plot=True)

    print(f"Head complexity: {evaluator.get_head_complexity()}")

    #accuracy = experiment.test_classifier(positive_test_files,
    #                                          negative_test_files)
        
    #print(accuracy)


    # evaluator_termination = AttentionEvaluatorClassifier(
    #     args.term_plot_dir,
    #     args.term_classifier_dir
    # )

    # evaluator_termination.add_true_from_files(termination_positive_files)
    # evaluator_termination.add_false_from_files(termination_negative_files)

    # evaluator_termination.evaluate_attentions(10)