import argparse
import random
import time
import os

import numpy as np
import ray
import torch
from tqdm import tqdm
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import Repeater
from ray.tune.schedulers import ASHAScheduler

from experiments import experiment
from portable.option.memory.set_dataset import SetDataset
from portable.utils.utils import load_gin_configs
from experiments.divdis_monte.core.divdis_monte_hyperparam_search_experiment import \
    MonteDivDisHyperparamSearchExperiment



img_dir = "/home/bingnan/portable_options/resources/monte_images/"
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
room_list = [0,4,3,9,8,10,11,5,13,7,6,14,22,21,19,18]

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
                         img_dir+"climb_down_ladder_room10_termination_positive.npy",
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
                         img_dir+"climb_down_ladder_room22_termination_positive.npy",
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
                       img_dir+"climb_down_ladder_room10_termination_positive.npy",
                       img_dir+"climb_down_ladder_room19_termination_positive.npy",
                       img_dir+"climb_down_ladder_room21_termination_positive.npy",
                       img_dir+"climb_down_ladder_room22_termination_positive.npy",
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
    #parser.add_argument("--config_file", nargs='+', type=str, required=True)
    #parser.add_argument("--gin_bindings", default=[], help='Gin bindings to override the values' + 
    #    ' set in the config files (e.g. "DQNAgent.epsilon_train=0.1",' +
    #    ' "create_atari_environment.game_name="Pong"").')

    args = parser.parse_args()

    #load_gin_configs(args.config_file, args.gin_bindings)


    experiment = MonteDivDisHyperparamSearchExperiment(experiment_name="minigrid_hyperparam_search",
                                                       base_dir=args.base_dir,
                                                       use_gpu=True)



    search_space = {
        "lr": tune.loguniform(1e-6, 1e-1),
        "l2_reg": tune.loguniform(1e-6, 1e-1),
        "div_weight":  tune.loguniform(1e-6, 1e-1),
        "num_heads": tune.randint(1, 12),
        "initial_epochs": tune.randint(50, 1000), # 50, 1000
        "epochs_per_room": tune.randint(10, 100), # 10, 100
        "unlabelled_batch_size": tune.choice([None, 16, 32, 64, 128, 256]),
    }

    #scheduler = ASHAScheduler(max_t=1000, grace_period=10, reduction_factor=2)
    optuna_search = OptunaSearch(metric=["best_weighted_acc", "num_heads"], mode=["max", "min"])
    #re_search_alg = Repeater(optuna_search, repeat=3)

    train_dataset = SetDataset(max_size=1e6, batchsize=32, unlabelled_batchsize=None)
    #train_dataset.add_true_files(positive_train_files)
    #train_dataset.add_false_files(negative_train_files)
    #train_dataset.add_unlabelled_files(initial_unlabelled_train_files)

    test_dataset_positive = SetDataset(max_size=1e6, batchsize=64, unlabelled_batchsize=None)
    test_dataset_positive.add_true_files(positive_test_files)
    test_dataset_negative = SetDataset(max_size=1e6, batchsize=64, unlabelled_batchsize=None)
    test_dataset_negative.add_false_files(negative_test_files)

    uncertain_dataset = SetDataset(max_size=1e6, batchsize=64, unlabelled_batchsize=None)
    uncertain_dataset.add_true_files(uncertain_test_files)
    

    #ray.init(max_direct_call_object_size=1024 ** 3)  # 1 GB
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                experiment.train_classifier, 
                train_dataset=train_dataset,
                positive_train_files=positive_train_files,
                negative_train_files=negative_train_files,
                unlabelled_train_files=initial_unlabelled_train_files,

                room_list=room_list,
                unlabelled_list=unlabelled_train_files,
                
                test_dataset_positive=test_dataset_positive,
                test_dataset_negative=test_dataset_negative,
                uncertain_dataset=uncertain_dataset,
                ),
            resources={"gpu":1}
        ),
        tune_config=tune.TuneConfig(
        search_alg=optuna_search,
        num_samples=5,
    ),
        param_space=search_space,
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("best_weighted_acc", "max")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final train loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final test weighted accuracy: {}".format(
        best_result.metrics["best_weighted_acc"]))
    print("Best trial final test raw accuracy: {}".format(
        best_result.metrics["best_acc"]))

