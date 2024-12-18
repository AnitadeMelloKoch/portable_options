import argparse
import os
import random
import time
import warnings

import numpy as np
import glob
import torch
from tqdm import tqdm
import transformers
#import torch_tensorrt

from portable.utils.utils import load_gin_configs
from experiments.classifier.core.classifier_experiment import DivDisClassifierExperiment
from experiments.classifier.core.utils import create_task_dict, get_data_path, filter_valid_images



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--config_file", nargs='+', type=str, required=True)
    parser.add_argument("--gin_bindings", default=[], help='Gin bindings to override the values' + 
        ' set in the config files (e.g. "DQNAgent.epsilon_train=0.1",' +
        ' "create_atari_environment.game_name="Pong"").')

    args = parser.parse_args()
    load_gin_configs(args.config_file, args.gin_bindings)

    warnings.filterwarnings("ignore", category=UserWarning, module='torch.nn.modules.lazy')
    transformers.logging.set_verbosity_error()  # This will show only errors, not warnings
    

    #torch.set_float32_matmul_precision('high')
    #torch.backends.cuda.matmul.allow_tf32 = True
    #torch.set_float32_matmul_precision('medium')
    #torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    #print(torch._dynamo.list_backends())
    seeds = [args.seed + i for i in range(args.n)]

    best_total_acc = []
    best_weighted_acc = []
    avg_weighted_acc = []
    train_time = []
    test_time = []

    dataset_path = 'resources/jaco/handle'

    for i in tqdm(range(args.n)):
        t0 = time.time()
        cur_seed = seeds[i]

        task_dict = create_task_dict(dataset_path)
        tasks = list(task_dict.keys())
        print(f"Tasks: {tasks}")
        
        train_tasks = ['microwave']
        for t in train_tasks:
            tasks.remove(t)
        test_tasks = tasks
        print(f"Train Tasks: {train_tasks}")
        print(f"Test Tasks: {test_tasks}")

        train_positive_files, train_negative_files,_,_ = get_data_path(dataset_path, task_dict, train_tasks, 'term')
        test_positive_files, test_negative_files,_,_ = get_data_path(dataset_path, task_dict, test_tasks, 'term')
        
        unlabelled_train_files = []
        #unlabelled_train_files += test_positive_files + test_negative_files
        #unlabelled_train_files = glob.glob('resources/jaco/stack/all_images/*.png') + glob.glob('resources/jaco/push/all_images/*.png')
        push_stack_files = glob.glob('resources/jaco/stack/all_images/*.png') + glob.glob('resources/jaco/push/all_images/*.png')
        unlabelled_train_files += filter_valid_images(push_stack_files)
        unlabelled_train_files = random.sample(unlabelled_train_files, int(0.7*len(unlabelled_train_files)))

        #corrupted_files = set(push_stack_files) - set(unlabelled_train_files)
        #if corrupted_files:
        #    for file in corrupted_files:
        #        print(f"Corrupted image found and skipped: {file}")
                
        print(f"Train Positive Files: {len(train_positive_files)}")
        print(f"Train Negative Files: {len(train_negative_files)}")
        print(f"Unlabelled Train Files: {len(unlabelled_train_files)}")
        print(f"Test Positive Files: {len(test_positive_files)}")
        print(f"Test Negative Files: {len(test_negative_files)}")

        experiment = DivDisClassifierExperiment(
                            base_dir=args.base_dir,
                            seed=cur_seed)
        experiment.add_datafiles(train_positive_files,
                    train_negative_files,
                    unlabelled_train_files)

        experiment.train_classifier()
        
        t1 = time.time()
        train_time.append(t1-t0)
        t2 = time.time()
        accuracy = experiment.test_classifier(test_positive_files, test_negative_files)
        test_time.append(time.time()-t2)
        
        print(f"Total Accuracy:    {np.round(accuracy[0], 2)}")
        print(f"Weighted Accuracy: {np.round(accuracy[1], 2)}")
        best_head = np.argmax(accuracy[1])
        best_weighted_acc.append(accuracy[1][best_head])
        best_total_acc.append(accuracy[0][best_head])
        avg_weighted_acc.append(np.mean(accuracy[1]))

    print(f"Best Total Accuracy:    {np.mean(best_total_acc):.2f}, {np.std(best_total_acc):.2f}")
    print(f"Best Weighted Accuracy: {np.mean(best_weighted_acc):.2f}, {np.std(best_weighted_acc):.2f}")
    print(f"Avg Weighted Accuracy:  {np.mean(avg_weighted_acc):.2f}, {np.std(avg_weighted_acc):.2f}")
    print(f"Train Time: {np.mean(train_time):.1f}, {np.std(train_time):.1f}")
    print(f"Test Time:  {np.mean(test_time):.1f}, {np.std(test_time):.1f}")

    