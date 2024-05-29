import argparse
import random
import time

import numpy as np
import torch
from tqdm import tqdm
#import torch_tensorrt

from portable.utils.utils import load_gin_configs
from experiments.divdis_minigrid.core.advanced_minigrid_divdis_classifier_experiment import \
    AdvancedMinigridDivDisClassifierExperiment


minigrid_positive_files = [
    ["resources/minigrid_images/adv_doorkey_8x8_v2_getbluekey_doorblue_0_termination_positive.npy"],
    ["resources/minigrid_images/adv_doorkey_8x8_v2_getgreenkey_doorgreen_0_termination_positive.npy"],
    ["resources/minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_0_termination_positive.npy"],
    ["resources/minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_0_termination_positive.npy"],
    ["resources/minigrid_images/adv_doorkey_8x8_v2_togoal_0_termination_positive.npy"],
    
]
minigrid_negative_files = [
    ["resources/minigrid_images/adv_doorkey_8x8_v2_getbluekey_doorblue_0_termination_negative.npy"],
    ["resources/minigrid_images/adv_doorkey_8x8_v2_getgreenkey_doorgreen_0_termination_negative.npy"],
    ["resources/minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_0_termination_negative.npy"],
    ["resources/minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_0_termination_negative.npy"],
    ["resources/minigrid_images/adv_doorkey_8x8_v2_togoal_0_termination_negative.npy"],
]
minigrid_unlabelled_files = [
    ["resources/minigrid_images/adv_doorkey_8x8_v2_getbluekey_doorblue_1_termination_positive.npy",
    "resources/minigrid_images/adv_doorkey_8x8_v2_getbluekey_doorblue_1_termination_negative.npy",
    "resources/minigrid_images/adv_doorkey_8x8_v2_getgreenkey_doorgreen_2_termination_positive.npy",
    "resources/minigrid_images/adv_doorkey_8x8_v2_getgreenkey_doorgreen_2_termination_negative.npy",],
    ["resources/minigrid_images/adv_doorkey_8x8_v2_getgreenkey_doorgreen_1_termination_positive.npy",
    "resources/minigrid_images/adv_doorkey_8x8_v2_getgreenkey_doorgreen_1_termination_negative.npy",
    "resources/minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_2_termination_positive.npy",
    "resources/minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_2_termination_negative.npy",],
    ["resources/minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_1_termination_positive.npy",
    "resources/minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_1_termination_negative.npy",
    "resources/minigrid_images/adv_doorkey_8x8_v2_getgreenkey_doorgreen_2_termination_positive.npy",
    "resources/minigrid_images/adv_doorkey_8x8_v2_getgreenkey_doorgreen_2_termination_negative.npy",],
    ["resources/minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_1_termination_positive.npy",
    "resources/minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_1_termination_negative.npy",
    "resources/minigrid_images/adv_doorkey_8x8_v2_getgreenkey_doorgreen_2_termination_positive.npy",
    "resources/minigrid_images/adv_doorkey_8x8_v2_getgreenkey_doorgreen_2_termination_negative.npy",],
    ["resources/minigrid_images/adv_doorkey_8x8_v2_togoal_1_termination_positive.npy",
    "resources/minigrid_images/adv_doorkey_8x8_v2_togoal_1_termination_negative.npy",
    "resources/minigrid_images/adv_doorkey_8x8_v2_getgreenkey_doorgreen_2_termination_positive.npy",
    "resources/minigrid_images/adv_doorkey_8x8_v2_getgreenkey_doorgreen_2_termination_negative.npy",]
]
minigrid_test_files_positive = [
    [
        "resources/minigrid_images/adv_doorkey_8x8_v2_getbluekey_doorblue_3_termination_positive.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getbluekey_doorblue_4_termination_positive.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getbluekey_doorblue_5_termination_positive.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getbluekey_doorblue_6_termination_positive.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getbluekey_doorblue_7_termination_positive.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getbluekey_doorblue_8_termination_positive.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getbluekey_doorblue_9_termination_positive.npy",
    ],
    [
        "resources/minigrid_images/adv_doorkey_8x8_v2_getgreenkey_doorgreen_3_termination_positive.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getgreenkey_doorgreen_4_termination_positive.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getgreenkey_doorgreen_5_termination_positive.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getgreenkey_doorgreen_6_termination_positive.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getgreenkey_doorgreen_7_termination_positive.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getgreenkey_doorgreen_8_termination_positive.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getgreenkey_doorgreen_9_termination_positive.npy",
    ],
    [
        "resources/minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_3_termination_positive.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_4_termination_positive.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_5_termination_positive.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_6_termination_positive.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_7_termination_positive.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_8_termination_positive.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_9_termination_positive.npy",
    ],
    [
        "resources/minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_3_termination_positive.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_4_termination_positive.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_5_termination_positive.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_6_termination_positive.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_7_termination_positive.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_8_termination_positive.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_9_termination_positive.npy",
    ],
    [
        "resources/minigrid_images/adv_doorkey_8x8_v2_togoal_3_termination_positive.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_togoal_4_termination_positive.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_togoal_5_termination_positive.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_togoal_6_termination_positive.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_togoal_7_termination_positive.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_togoal_8_termination_positive.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_togoal_9_termination_positive.npy",
    ],
]
minigrid_test_files_negative = [
    [
        "resources/minigrid_images/adv_doorkey_8x8_v2_getbluekey_doorblue_3_termination_negative.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getbluekey_doorblue_4_termination_negative.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getbluekey_doorblue_5_termination_negative.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getbluekey_doorblue_6_termination_negative.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getbluekey_doorblue_7_termination_negative.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getbluekey_doorblue_8_termination_negative.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getbluekey_doorblue_9_termination_negative.npy",
    ],
    [
        "resources/minigrid_images/adv_doorkey_8x8_v2_getgreenkey_doorgreen_3_termination_negative.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getgreenkey_doorgreen_4_termination_negative.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getgreenkey_doorgreen_5_termination_negative.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getgreenkey_doorgreen_6_termination_negative.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getgreenkey_doorgreen_7_termination_negative.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getgreenkey_doorgreen_8_termination_negative.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getgreenkey_doorgreen_9_termination_negative.npy",
    ],
    [
        "resources/minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_3_termination_negative.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_4_termination_negative.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_5_termination_negative.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_6_termination_negative.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_7_termination_negative.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_8_termination_negative.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_9_termination_negative.npy",
    ],
    [
        "resources/minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_3_termination_negative.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_4_termination_negative.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_5_termination_negative.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_6_termination_negative.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_7_termination_negative.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_8_termination_negative.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_9_termination_negative.npy",
    ],
    [
        "resources/minigrid_images/adv_doorkey_8x8_v2_togoal_3_termination_negative.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_togoal_4_termination_negative.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_togoal_5_termination_negative.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_togoal_6_termination_negative.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_togoal_7_termination_negative.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_togoal_8_termination_negative.npy",
        "resources/minigrid_images/adv_doorkey_8x8_v2_togoal_9_termination_negative.npy",
    ],
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

    #torch.set_float32_matmul_precision('high')
    #torch.backends.cuda.matmul.allow_tf32 = True
    #torch.set_float32_matmul_precision('medium')
    #torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    #print(torch._dynamo.list_backends())

    best_total_acc = []
    best_weighted_acc = []
    avg_weighted_acc = []


    for i in tqdm(range(len(minigrid_positive_files)-1)):
        positive_train_files = minigrid_positive_files[i]
        negative_train_files = minigrid_negative_files[i]
        unlabelled_train_files = minigrid_unlabelled_files[i]
        positive_test_files = minigrid_test_files_positive[i]
        negative_test_files = minigrid_test_files_negative[i]
        print(f'Now training on task {positive_train_files[0].split("/")[-1]}')
            
        experiment = AdvancedMinigridDivDisClassifierExperiment(
                            base_dir=args.base_dir,
                            seed=args.seed,)

        experiment.add_datafiles(positive_train_files,
                    negative_train_files,
                    unlabelled_train_files)

        experiment.train_classifier()
        

        accuracy = experiment.test_classifier(positive_test_files,
                            negative_test_files)
        
        print(f"Total Accuracy: {np.round(accuracy[0], 4)}")
        print(f"Weighted Accuracy: {np.round(accuracy[1], 4)}")
        best_head = np.argmax(accuracy[1])



    