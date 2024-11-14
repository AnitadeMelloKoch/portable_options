import os
import logging
import multiprocessing
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import gin

from pyexpat import model
from experiments.dog.core.dog_experiment import DogExperiment
from portable.utils.utils import load_gin_configs
from portable.option.memory import SetDataset, UnbalancedSetDataset
from portable.option.divdis.models.mlp import MultiHeadMLP, OneHeadMLP
from portable.option.divdis.models.minigrid_cnn_16x16 import MinigridCNN16x16, MinigridCNNLarge
from portable.option.divdis.models.minigrid_cnn import MinigridCNN
from portable.option.divdis.models.monte_cnn import MonteCNN
from portable.option.divdis.models.clip import Clip
from portable.option.divdis.divdis import DivDisLoss

import os
import random

# Define the directory containing the images
img_dir = "/oscar/data/gdk/yyang239/portable_options/resources/dog_images"
print(img_dir)
# List all the files in the directory
files = os.listdir(img_dir)

# Initialize empty lists to store filtered file paths
chihuahua_files = []
spaniel_files = []

# Filter the files based on breed
for file in files:
    file_path = os.path.join(img_dir, file)
    
    if 'chihuahua' in file.lower() and file.endswith('.npy'):
        chihuahua_files.append(file_path)
        
    elif 'spaniel' in file.lower() and file.endswith('.npy'):
        spaniel_files.append(file_path)

# Shuffle the files for random partitioning
random.shuffle(chihuahua_files)
random.shuffle(spaniel_files)

# Define partition percentages
def partition_data(files, label_percent, unlabel_percent, test_percent):
    total = len(files)
    
    # Compute the split indices
    label_count = int(total * label_percent)
    unlabel_count = int(total * unlabel_percent)
    
    # Partition the data
    labeled = files[:label_count]
    unlabeled = files[label_count:label_count + unlabel_count]
    test = files[label_count + unlabel_count:]
    
    return labeled, unlabeled, test

# Partition chihuahua files
chihuahua_train_labeled, chihuahua_train_unlabeled, chihuahua_test = partition_data(chihuahua_files, 0.2, 0.3, 0.5)

# Partition spaniel files
spaniel_train_labeled, spaniel_train_unlabeled, spaniel_test = partition_data(spaniel_files, 0.2, 0.3, 0.5)
unlabeled_data = chihuahua_train_unlabeled + spaniel_train_unlabeled

# Output the counts for each category
print(f"Chihuahua - Labeled: {len(chihuahua_train_labeled)}, Unlabeled: {len(chihuahua_train_unlabeled)}, Test: {len(chihuahua_test)}")
print(f"Spaniel - Labeled: {len(spaniel_train_labeled)}, Unlabeled: {len(spaniel_train_unlabeled)}, Test: {len(spaniel_test)}")
room_list = [0, 4, 3, 9, 8, 10, 11, 5, #12 here, has nothing
             13, 7, 6, 2, 14, 22, # 23 
             21, 19, 18]


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

        multiprocessing.set_start_method('spawn')

        seeds = [args.seed * i for i in range(1, 6)]
        room_histories = []
        additional_histories = []

        for seed in seeds:
            print(f"Running experiment for seed {seed}")
        
            experiment = DogExperiment(base_dir=args.base_dir,
                                        seed=seed,
                                        use_gpu=False)

            experiment.add_train_files(chihuahua_train_labeled,
                                       spaniel_train_labeled,
                                       unlabeled_data)
            experiment.add_test_files(chihuahua_test,
                                      spaniel_test)
            
            experiment.train_classifier(experiment.initial_epochs)

            print("Training on room 1 only")
            logging.info("Training on room 1 only")
            accuracy_pos, accuracy_neg, accuracy, weighted_acc = experiment.test_classifier()
            uncertainty = experiment.test_uncertainty()
                                                        
            print(f"Weighted Accuracy: {weighted_acc}")
            print(f"Accuracy: {accuracy}")
            print(f"Uncertainty: {uncertainty}")

            best_weighted_acc = np.max(weighted_acc)
            best_head_idx = np.argmax(weighted_acc)
            best_accuracy = accuracy[best_head_idx]
            best_true_acc = accuracy_pos[best_head_idx]
            best_false_acc = accuracy_neg[best_head_idx]
            best_head_uncertainty = uncertainty[best_head_idx]

            history = {
            'weighted_accuracy': [best_weighted_acc],
            'raw_accuracy': [best_accuracy],
            'true_accuracy': [best_true_acc], 
            'false_accuracy': [best_false_acc],
            'uncertainty': [best_head_uncertainty]
        }

            history = experiment.room_by_room_train(room_list, unlabeled_data, history)
            room_histories.append(history)

            print("All unlabelled rooms added, now running additional training loops")
            logging.info("All unlabelled rooms added, now running additional training loops")

            history = experiment.additional_train()
            additional_histories.append(history)
    
        experiment.plot_metrics(room_histories, 'room', 'avg_room_train_metrics')
        experiment.plot_metrics(additional_histories, 'additional train loops', 'avg_additional_train_metrics')
        
        #num_batch = 1
        #view_acc = experiment.view_false_predictions(positive_test_files, negative_test_files, num_batch)
        #print(f"Viewing {num_batch} of Predictions:")
        #print(f"Accuracy: {view_acc[0]}")
        #print(f"Weighted Accuracy: {view_acc[1]}")

