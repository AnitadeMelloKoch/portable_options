from cgi import test
from pathlib import Path
import logging
import argparse
import re

import numpy as np
import torch
from portable.utils.utils import load_gin_configs
from . import ClassifierExperiment
#from experiments.classifier.core.class_experiment import ClassifierExperiment


'''
To run this experiment, use the following command:
python -m experiments.classifier.minigrid_confidence_weighted_loss --task openpurpledoor_doorpurple initiation --base_dir experiments/classifier/ --exp_seed 0 --train_seed 0 --test_seed 1 2 --config_file configs/confidence_weighted_loss.gin
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", nargs='+', type=str, required=True, help='For openpurpledoor_doorpurple initiation set, write \'openpurpledoor_doorpurple initiation\'')

    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--exp_seed", type=int, required=True)
    parser.add_argument("--train_seed", nargs='+', type=int, required=True)
    parser.add_argument("--test_seed",  nargs='+', type=int, required=True)
    parser.add_argument("--config_file", nargs='+', type=str, required=True)
    parser.add_argument("--gin_bindings", default=[], help='Gin bindings to override the values' + 
	    ' set in the config files (e.g. "DQNAgent.epsilon_train=0.1",' +
	    ' "create_atari_environment.game_name="Pong"").')
    
    args = parser.parse_args()
    load_gin_configs(args.config_file, args.gin_bindings)

    task = args.task
    #task_name, init_or_term = re.split('[ ,]+', task)[:2]
    task_name, init_or_term = task
    
    def task_to_pattern(task_name, init_or_term, seed, positive_or_negative):
        pattern = rf'.*8x8.*{re.escape(task_name)}.*{re.escape(str(seed))}.*{re.escape(init_or_term)}.*{re.escape(positive_or_negative)}\.npy'
        return pattern

    dataset_folder = Path('resources/minigrid_images/')
    all_images = list(dataset_folder.glob('*.npy'))
    #   rf'{re.escape(A)}[ ,_]+{re.escape(B)}[ ,_]+{re.escape(C)}'
    #   adv_doorkey_8x8_getbluekey_doorblue_0_initiation_negative.npy
    train_pos_regex = []
    train_neg_regex = []
    test_pos_regex = []
    test_neg_regex = []
    for i in range(len(args.train_seed)):
        train_pos_regex.append(task_to_pattern(task_name, init_or_term, args.train_seed[i], 'positive'))
        train_neg_regex.append(task_to_pattern(task_name, init_or_term, args.train_seed[i], 'negative'))
    for i in range(len(args.test_seed)):
        test_pos_regex.append(task_to_pattern(task_name, init_or_term, args.test_seed[i], 'positive'))
        test_neg_regex.append(task_to_pattern(task_name, init_or_term, args.test_seed[i], 'negative'))
    
    train_pos_files = []
    train_neg_files = []
    test_pos_files = []
    test_neg_files = []
    for i in range(len(train_pos_regex)):
        train_pos_files.append([f for f in all_images if re.match(train_pos_regex[i], str(f))])
        train_neg_files.append([f for f in all_images if re.match(train_neg_regex[i], str(f))])
    for i in range(len(test_neg_regex)):
        test_pos_files.append([f for f in all_images if re.match(test_pos_regex[i], str(f))])
        test_neg_files.append([f for f in all_images if re.match(test_neg_regex[i], str(f))])  

    print("Train sets: ", len(train_pos_files))
    print("Test sets: ", len(test_pos_files))
	
    experiment = ClassifierExperiment(base_dir=args.base_dir,
				                      seed=args.exp_seed)

    logging.info(f'===EXPERIMENT PARAMS===')
    logging.info(f' --TASK: {task_name} ({init_or_term})')
    logging.info(f' --EXPERIMENT SEED: {args.exp_seed}')
    logging.info(f' --TRAIN SEEDS: {args.train_seed}')
    logging.info(f' --TEST SEEDS: {args.test_seed}')
    logging.info(f' --CONFIDENCE: NOT updating when training on test sets')
    logging.info(f' --SD cutoff: 4')
    logging.info(f' --NOTE: Using progressive training.')
    #logging.info(f' --NOTE: This is not test w/ perfect test labels. Not progressive training.')

    
    print("***Adding Train & Val Images***") # train sets always all added at once
    for i in range(len(train_pos_files)):
        experiment.add_train_data(train_pos_files[i], train_neg_files[i])
    """print("TRAIN positive img: ", len(experiment.classifier.dataset.true_data))
    print("TRAIN false img: ", len(experiment.classifier.dataset.false_data))
    print("VAL positive img: ", len(experiment.classifier.dataset.validate_indicies_true))
    print("VAL false img: ", len(experiment.classifier.dataset.validate_indicies_false))"""


    print("***Adding Testsets Data***")
    for i in range(len(test_pos_files)):
        experiment.add_test_data(i, test_pos_files[i], test_neg_files[i])

    
    print("***Begin training classifier***")
    experiment.train(True)
    print("Classifier trained.")
    logging.info('classifier trained')

    print("==================VALIDATION 0==================")
    logging.info("==================VALIDATION 0==================")
    print("After train on set 0, check val accuracy")

    val_true = experiment.classifier.dataset.true_data[experiment.classifier.dataset.validate_indicies_true]
    val_false = experiment.classifier.dataset.false_data[experiment.classifier.dataset.validate_indicies_false]
    logging.info(f'Val True shape : {val_true.shape}')
    logging.info(f'Val False shape: {val_false.shape}')

    #results = experiment.predict(val_true)
    experiment.val_accuracy(val_true, val_false)


    print("==================VALIDATION 1==================")
    logging.info("==================VALIDATION 1==================")

    test_set1 = np.concatenate((experiment.test_sets[0].true_data, experiment.test_sets[0].false_data))
    labels1 = np.concatenate((np.array([1]*len(experiment.test_sets[0].true_data)), np.array([0]*len(experiment.test_sets[0].false_data))))

    experiment.progressive_train(test_set1, torch.from_numpy(labels1))
    #experiment.progressive_train_testing(test_pos_files[0], test_neg_files[0], test_set1, labels1)
    
    experiment.val_accuracy(val_true, val_false)

    print("==================VALIDATION 2==================")
    logging.info("==================VALIDATION 2==================")
    test_set2 = np.concatenate((experiment.test_sets[1].true_data, experiment.test_sets[1].false_data))
    labels2 = np.concatenate((np.array([1]*len(experiment.test_sets[1].true_data)), np.array([0]*len(experiment.test_sets[1].false_data))))

    experiment.progressive_train(test_set2, torch.from_numpy(labels2))
    #experiment.progressive_train_testing(test_pos_files[1], test_neg_files[1], test_set2, labels2)

    experiment.val_accuracy(val_true, val_false)
    
    