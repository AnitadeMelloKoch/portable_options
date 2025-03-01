import argparse 
from datetime import datetime
import os
import random
import warnings
import glob
import transformers
import numpy as np
import torch
from tqdm import tqdm
import transformers

from portable.utils.utils import load_gin_configs
from experiments.classifier.core.classifier_sweep_experiment import DivDisClassifierSweepExperiment
from experiments.classifier.core.utils import create_task_dict, get_data_path, filter_valid_images



def formatted_time():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

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

    warnings.filterwarnings("ignore", category=UserWarning, module='torch.nn.modules.lazy')
    transformers.logging.set_verbosity_error()  # This will show only errors, not warnings


    dataset_path = 'resources/jaco/handle'
    
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
    push_stack_files = glob.glob('resources/jaco/stack/all_images/*.png') + glob.glob('resources/jaco/push/all_images/*.png')
    unlabelled_train_files += filter_valid_images(push_stack_files)
    unlabelled_train_files = random.sample(unlabelled_train_files, int(0.4*len(unlabelled_train_files)))
            
    print(f"Train Positive Files: {len(train_positive_files)}")
    print(f"Train Negative Files: {len(train_negative_files)}")
    print(f"Unlabelled Train Files: {len(unlabelled_train_files)}")
    print(f"Test Positive Files: {len(test_positive_files)}")
    print(f"Test Negative Files: {len(test_negative_files)}")

    experiment = DivDisClassifierSweepExperiment(base_dir=args.base_dir,
                                            train_positive_files=train_positive_files,
                                            train_negative_files=train_negative_files,
                                            unlabelled_files=unlabelled_train_files,
                                            test_positive_files=test_positive_files,
                                            test_negative_files=test_negative_files,
                                            seed=args.seed)
        
    NUM_SEEDS = 5


    '''print(f"[{formatted_time()}] Now running grid search...")
    experiment.grid_search(lr_range=np.logspace(-4, -3, 3),
                            div_weight_range=np.logspace(-4, -2, 4),
                            l2_reg_range=np.logspace(-4, -2, 3),
                            head_num_range=[4,6,8],
                            epochs_range=[30], #[30,70,150,300]
                            num_seeds=NUM_SEEDS)'''

    print(f"[{formatted_time()}] Sweeping learning rate...")
    experiment.sweep_lr(-4, # 0.0001
                        -2,
                        8,
                        NUM_SEEDS)
    
    print(f"[{formatted_time()}] Sweeping class div weight...")
    experiment.sweep_class_div_weight(-7, # 0.0000001
                                      0,
                                      8,
                                      NUM_SEEDS)

    print(f"[{formatted_time()}] Sweeping L2 reg weight...")
    experiment.sweep_l2_reg_weight(-5, # 0.000001
                                   -2,
                                   8,
                                   NUM_SEEDS)


    print(f"[{formatted_time()}] Sweeping ensemble size...")
    experiment.sweep_ensemble_size(1, 
                                   10,
                                   2,
                                   NUM_SEEDS,
                                   [1,2,3,5,7,10]
                                   )

    
    print(f"[{formatted_time()}] Sweeping epochs...")
    experiment.sweep_epochs(5, 
                            100, 
                            10,
                            NUM_SEEDS,
                            [5,10,15,25,50]) # when a list is provided, use this

    print(f"[{formatted_time()}] Sweeping class weights...")
    experiment.sweep_class_weights(0.1, 
                                   .95, 
                                   7,
                                   NUM_SEEDS)


    print(f"[{formatted_time()}] Sweeping div batch size...")
    experiment.sweep_div_batch_size(16,
                                    400,
                                    16,
                                    NUM_SEEDS,
                                    [16,32,64,128])




