import argparse
import re
#from experiments.classifier.core.class_experiment import ClassifierExperiment
from . import ClassifierExperiment
from portable.utils.utils import load_gin_configs
from pathlib import Path
import numpy as np


# TODO: set up arg parser and config file, if needed
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
    
    train_positive_files = []
    train_negative_files = []
    test_positive_files = []
    test_pnegative_files = []

    train_positive_regex = r'.*8x8.*_[0]_.*positive.*\.npy$'
    train_negative_regex = r'.*8x8.*_[0]_.*negative.*\.npy$'
    test_positive_regex = r'.*8x8.*_[12]_.*positive.*\.npy$'
    test_negative_regex = r'.*8x8.*_[12]_.*negative.*\.npy$'

    dataset_folder = Path('resources/minigrid_images/')
    all_images = list(dataset_folder.glob('*.npy'))

    train_positive_files = [f for f in all_images if re.match(train_positive_regex, str(f))]
    train_negative_files = [f for f in all_images if re.match(train_negative_regex, str(f))]
    test_positive_files = [f for f in all_images if re.match(test_positive_regex, str(f))]
    test_pnegative_files = [f for f in all_images if re.match(test_negative_regex, str(f))]

    print("Train positive files: ", len(train_positive_files))
    print("Train negative files: ", len(train_negative_files))
    print("Test positive files: ", len(test_positive_files))
    print("Test negative files: ", len(test_pnegative_files))
	
    experiment = ClassifierExperiment(base_dir=args.base_dir,
				      seed=args.seed,
				      experiment_name='sample confidence test -- minigrid',
				      classifier_train_epochs=100,
				      use_gpu=True)
    print("Adding train data...")
    experiment.add_train_data(train_positive_files, train_negative_files)
    print("Positive data length: ", len(experiment.classifier.dataset.true_data))
    print("False data length: ", len(experiment.classifier.dataset.false_data))
    print("Priority false data length: ", len(experiment.classifier.dataset.priority_false_data))

    print("Adding test data...")
    experiment.add_test_data(test_positive_files, test_pnegative_files)
    print("Positive data length: ", len(experiment.test_dataset.true_data))
    print("False data length: ", len(experiment.test_dataset.false_data))
    print("Priority false data length: ", len(experiment.test_dataset.priority_false_data))
    
    print("Begin training...")
    experiment.train()
    print("Classifier trained.")
    
    print("Begin stats...")
    experiment.stats_experiment()
    print("Stats experiment complete.")