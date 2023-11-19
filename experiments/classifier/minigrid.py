import argparse
import re
#from classifier.core.classifier_experiment import ClassifierExperiment
from . import ClassifierExperiment
from portable.utils.utils import load_gin_configs
from pathlib import Path
import numpy as np

'''
# TODO: set up arg parser and config file, if needed
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--skip_train_options", action='store_true')
    parser.add_argument("--config_file", nargs='+', type=str, required=True)
    parser.add_argument("--gin_bindings", default=[], help='Gin bindings to override the values' + 
            ' set in the config files (e.g. "DQNAgent.epsilon_train=0.1",' +
            ' "create_atari_environment.game_name="Pong"").')
    
    args = parser.parse_args()
    
    load_gin_configs(args.config_file, args.gin_bindings)
'''

load_gin_configs('~/portable_options/configs/classifier.gin', [])
  
train_positive_files = []
train_negative_files = []
test_positive_files = []
test_pnegative_files = []

train_positive_regex = r'.*_[0]_.*positive.*\.npy$'
train_negative_regex = r'.*_[0]_.*negative.*\.npy$'
test_positive_regex = r'.*_[12]_.*positive.*\.npy$'
test_negative_regex = r'.*_[12]_.*negative.*\.npy$'

dataset_folder = Path('~/portable_options/resources/minigrid_images/').expanduser()
all_images = dataset_folder.glob('*.npy')

train_positive_files = [np.load(f) for f in all_images if re.match(train_positive_regex, str(f))]
train_negative_files = [np.load(f) for f in all_images if re.match(train_negative_regex, str(f))]
test_positive_files = [np.load(f) for f in all_images if re.match(test_positive_regex, str(f))]
test_pnegative_files = [np.load(f) for f in all_images if re.match(test_negative_regex, str(f))]


experiment = ClassifierExperiment(base_dir='experiments/classifier/minigrid',
                                  seed=0,
                                  experiment_name='sample confidence test -- minigrid',
                                  classifier_train_epochs=100,
                                  use_gpu=True)

experiment.add_train_data(train_positive_files, train_negative_files)
experiment.add_test_data(test_positive_files, test_pnegative_files)

experiment.stats_experiment()