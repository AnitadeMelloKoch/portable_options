import argparse
import pickle
import random
import time
import os

import numpy as np
import ray
import torch
from tqdm import tqdm
from ray import tune, train
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import Repeater
from ray.tune.schedulers import ASHAScheduler

from experiments import experiment
from portable.option.memory.set_dataset import SetDataset
from portable.utils.utils import load_gin_configs
from experiments.divdis_minigrid.core.advanced_minigrid_divdis_hyperparam_search_experiment import \
    AdvancedMinigridDivDisHyperparamSearchExperiment


color = 'grey'
task = f'get{color}key'
#task = f'open{color}door'
init_term = 'termination'
RANDOM_TRAIN = True
RANDOM_UNLABELLED = True

base_img_dir = 'resources/minigrid_images/'
positive_train_files = [f"{base_img_dir}adv_doorkey_8x8_v2_{task}_door{color}_0_{init_term}_positive.npy"]
negative_train_files = [f"{base_img_dir}adv_doorkey_8x8_v2_{task}_door{color}_0_{init_term}_negative.npy"]
unlabelled_train_files = [f"{base_img_dir}adv_doorkey_8x8_v2_{task}_door{color}_{s}_{init_term}_{pos_neg}.npy" for s in [1,2] for pos_neg in ['positive', 'negative']]
positive_test_files = [f"{base_img_dir}adv_doorkey_8x8_v2_{task}_door{color}_{s}_{init_term}_positive.npy" for s in [5,6,7,8,9,10]]
negative_test_files = [f"{base_img_dir}adv_doorkey_8x8_v2_{task}_door{color}_{s}_{init_term}_negative.npy" for s in [5,6,7,8,9,10]]

if RANDOM_TRAIN:
    positive_train_files += [f"{base_img_dir}adv_doorkey_8x8_v2_{task}_door{color}_0_1_{init_term}_positive.npy"]
    negative_train_files += [f"{base_img_dir}adv_doorkey_8x8_v2_{task}_door{color}_0_1_{init_term}_negative.npy"]
if RANDOM_UNLABELLED:
    unlabelled_train_files += [f"{base_img_dir}adv_doorkey_8x8_v2_{task}_door{color}_{s}_1_{init_term}_{pos_neg}.npy" for s in [1,2] for pos_neg in ['positive', 'negative']]
positive_test_files += [f"{base_img_dir}adv_doorkey_8x8_v2_{task}_door{color}_{s}_1_{init_term}_positive.npy" for s in [5,6,7,8,9,10,11]]
negative_test_files += [f"{base_img_dir}adv_doorkey_8x8_v2_{task}_door{color}_{s}_1_{init_term}_negative.npy" for s in [5,6,7,8,9,10,11]]




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_dir", type=str, required=True)
    #parser.add_argument("--config_file", nargs='+', type=str, required=True)
    #parser.add_argument("--gin_bindings", default=[], help='Gin bindings to override the values' + 
    #    ' set in the config files (e.g. "DQNAgent.epsilon_train=0.1",' +
    #    ' "create_atari_environment.game_name="Pong"").')

    args = parser.parse_args()

    #load_gin_configs(args.config_file, args.gin_bindings)



    experiment = AdvancedMinigridDivDisHyperparamSearchExperiment(experiment_name="minigrid_hyperparam_search",
                                                                  base_dir=args.base_dir,
                                                                  train_positive_files=positive_train_files,
                                                                  train_negative_files=negative_train_files,
                                                                  unlabelled_files=unlabelled_train_files,
                                                                  test_positive_files=positive_test_files,
                                                                  test_negative_files=negative_test_files)

    ray.init()


    search_space = {
        "lr": tune.loguniform(1e-5, 1e-2),
        "l2_reg": tune.loguniform(1e-5, 1e-1),
        "div_weight":  tune.loguniform(1e-5, 1e-1),
        "num_heads": tune.randint(1, 6),
        "num_epochs": tune.randint(50, 1000),
        "unlabelled_batch_size": tune.choice([16, 32, 64, 128, 256]),
    }

    #scheduler = ASHAScheduler(max_t=1000, grace_period=10, reduction_factor=2)
    optuna_search = OptunaSearch(metric="best_weighted_acc", mode="max")
    #optuna_search = OptunaSearch(metric="best_weighted_acc", mode="max")

    re_search_alg = Repeater(optuna_search, repeat=3)

    train_dataset = SetDataset(max_size=1e6, batchsize=32, unlabelled_batchsize=None)
    train_dataset.add_true_files(positive_train_files)
    train_dataset.add_false_files(negative_train_files)
    train_dataset.add_unlabelled_files(unlabelled_train_files)

    test_dataset_positive = SetDataset(max_size=1e6, batchsize=64, unlabelled_batchsize=None)
    test_dataset_positive.add_true_files(positive_test_files)
    test_dataset_negative = SetDataset(max_size=1e6, batchsize=64, unlabelled_batchsize=None)
    test_dataset_negative.add_false_files(negative_test_files)

    #ray.init(max_direct_call_object_size=1024 ** 3)  # 1 GB
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                experiment.train_classifier, 
                train_dataset=train_dataset,
                test_dataset_positive=test_dataset_positive,
                test_dataset_negative=test_dataset_negative),
            resources={"cpu":16, "gpu":1}
        ),
        tune_config=tune.TuneConfig(
        search_alg=re_search_alg,
        num_samples=120,
        max_concurrent_trials=1,
    ),
        param_space=search_space,
    )
    results = tuner.fit()

    with open(os.path.join(experiment.log_dir, results), 'wb') as f:
        pickle.dump("minigrid_hyperparam_search_results", f)
    
    best_result = results.get_best_result("best_weighted_acc", "max")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final train loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final test weighted accuracy: {}".format(
        best_result.metrics["best_weighted_acc"]))
    print("Best trial final test raw accuracy: {}".format(
        best_result.metrics["best_acc"]))

