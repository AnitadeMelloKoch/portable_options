import argparse
import re
#from experiments.classifier.core.class_experiment import ClassifierExperiment
from . import ClassifierExperiment
from portable.utils.utils import load_gin_configs
from pathlib import Path
import numpy as np

'''
To run this experiment, use the following command:
python -m experiments.classifier.minigrid_confidence_weighted_loss --task 'blue red initiation' --base_dir experiments/classifier/ --seed 0 --config_file configs/confidence_weighted_loss.gin
'''

# TODO: set up arg parser and config file, if needed
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, required=True, help='For getbluekey_doorred initiation set, write \'blue red initiation\'')

    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config_file", nargs='+', type=str, required=True)
    parser.add_argument("--gin_bindings", default=[], help='Gin bindings to override the values' + 
	    ' set in the config files (e.g. "DQNAgent.epsilon_train=0.1",' +
	    ' "create_atari_environment.game_name="Pong"").')
    
    args = parser.parse_args()
    load_gin_configs(args.config_file, args.gin_bindings)

    task = args.task
    key_color, door_color, init_or_term = re.split('[ ,_]+', task)[:3]
    
    def task_to_pattern(key_color, door_color, init_or_term, seed, positive_or_negative):
        pattern = rf'8x8.*{re.escape(key_color)}.*door{re.escape(door_color)}.*[{re.escape(str(seed))}].*{re.escape(init_or_term)}.*{re.escape(positive_or_negative)}\.npy'
        return pattern


    dataset_folder = Path('resources/minigrid_images/')
    all_images = list(dataset_folder.glob('*.npy'))
    #   rf'{re.escape(A)}[ ,_]+{re.escape(B)}[ ,_]+{re.escape(C)}'
    #   adv_doorkey_8x8_getbluekey_doorblue_0_initiation_negative.npy
    train_positive_regex = task_to_pattern(key_color, door_color, init_or_term, 0, 'positive')
    train_negative_regex = task_to_pattern(key_color, door_color, init_or_term, 0, 'negative')
    test_positive_regex = task_to_pattern(key_color, door_color, init_or_term, 12, 'positive')
    test_negative_regex = task_to_pattern(key_color, door_color, init_or_term, 12, 'negative')

    train_positive_files = [f for f in all_images if re.match(train_positive_regex, str(f))]
    train_negative_files = [f for f in all_images if re.match(train_negative_regex, str(f))]
    test_positive_files = [f for f in all_images if re.match(test_positive_regex, str(f))]
    test_pnegative_files = [f for f in all_images if re.match(test_negative_regex, str(f))]

    #print("Train positive files: ", len(train_positive_files))
    #print("Train negative files: ", len(train_negative_files))
    #print("Test positive files: ", len(test_positive_files))
    #print("Test negative files: ", len(test_pnegative_files))
	
    experiment = ClassifierExperiment(base_dir=args.base_dir,
				                      seed=args.seed)
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