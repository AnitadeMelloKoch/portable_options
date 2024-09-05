import argparse
import os
import random

import torch

#from evaluators import DivDisEvaluatorClassifier
from evaluators import DivDisEvaluatorClassifier
#from experiments.divdis_minigrid.core.advanced_minigrid_factored_divdis_classifier_experiment import \
#    AdvancedMinigridFactoredDivDisClassifierExperiment
from portable.option.divdis.divdis_classifier import DivDisClassifier
from portable.utils.utils import load_gin_configs


color = 'grey'
task = f'open{color}door'
init_term = 'initiation'

base_img_dir = 'resources/minigrid_images/'
positive_train_files = [f"{base_img_dir}adv_doorkey_8x8_v2_{task}_door{color}_0_{init_term}_positive.npy"]
negative_train_files = [f"{base_img_dir}adv_doorkey_8x8_v2_{task}_door{color}_0_{init_term}_negative.npy"]
unlabelled_train_files = [f"{base_img_dir}adv_doorkey_8x8_v2_{task}_door{color}_{s}_{init_term}_{pos_neg}.npy" for s in [1,2] for pos_neg in ['positive', 'negative']]
positive_test_files = [f"{base_img_dir}adv_doorkey_8x8_v2_{task}_door{color}_{s}_{init_term}_positive.npy" for s in [3,4,5,6,7,8,9,10]]
negative_test_files = [f"{base_img_dir}adv_doorkey_8x8_v2_{task}_door{color}_{s}_{init_term}_negative.npy" for s in [3,4,5,6,7,8,9,10]]


'''
positive_train_files = ["resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_0_initiation_positive.npy"]

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

'''

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
'''


def factored_transform(x):
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
                    image_input=True,
                    batch_size=32,
                    base_dir=args.base_dir)




    evaluator.add_test_files(positive_test_files, negative_test_files)
    #evaluator.test_dataset.set_transform_function(factored_transform)

    evaluator.evaluate(1.0)

    # print head complexity
    print(evaluator.get_head_complexity())