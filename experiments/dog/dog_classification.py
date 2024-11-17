import logging
import multiprocessing
from itertools import chain
import os
from experiments.dog.core.dog_experiment import DogExperiment
import argparse 
from portable.utils.utils import load_gin_configs
import random
import numpy as np

# img_dir = "/oscar/data/gdk/yyang239/portable_options/resources/dog_images"
img_dir = "/home/yyang239/divdis/portable_options/resources/dog_images"
# Loop through all files in the directory
chihuahua = []
spaniel = []
for filename in os.listdir(img_dir):
    # Check if the file starts with 'chihuahua_'
    if filename.startswith('chihuahua_'):
        # positive
        chihuahua.append(img_dir+'/'+filename)
    else:
        # negative
        spaniel.append(img_dir+'/'+filename)

# Function to split data into 20%, 60%, and 20%
def split_data(data_list):
    # Shuffle the data
    random.shuffle(data_list)
    
    # Split into 20%, 60%, and 20%
    n = len(data_list)
    split_20 = int(0.2 * n)
    split_60 = int(0.6 * n)
    
    # Create splits
    unlabeled_data = data_list[:split_20]
    train_data = data_list[split_20:split_20 + split_60]
    test_data = data_list[split_20 + split_60:]
    
    return unlabeled_data, train_data, test_data

# Split the data for chihuahua and spaniel
unlabeled_chihuahua, train_chihuahua, test_chihuahua = split_data(chihuahua)
unlabeled_spaniel, train_spaniel, test_spaniel = split_data(spaniel)

# Combine the unlabeled data from both categories
unlabeled_data = unlabeled_chihuahua + unlabeled_spaniel

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
                                                            use_gpu = True)

            experiment.add_train_files(train_chihuahua,
                                       train_spaniel,
                                       unlabeled_data)
            experiment.add_test_files(test_chihuahua,
                                      test_spaniel)
            
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