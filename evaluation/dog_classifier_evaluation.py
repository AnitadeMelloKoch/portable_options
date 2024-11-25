import argparse
import os
import random
import re
import warnings
import torch
#from evaluators import DivDisEvaluatorClassifier
from evaluation.evaluators.divdis_evaluator_classifier import DivDisEvaluatorClassifier
#from experiments.divdis_minigrid.core.advanced_minigrid_factored_divdis_classifier_experiment import \
#    AdvancedMinigridFactoredDivDisClassifierExperiment
from portable.option.divdis.divdis_classifier import DivDisClassifier
from portable.utils.utils import load_gin_configs, set_seed
from experiments.dog.dog_classification import unlabeled_chihuahua, train_chihuahua, test_chihuahua, unlabeled_spaniel, train_spaniel, test_spaniel, unlabeled_data
img_dir = "resources/dog_images/"
def get_sorted_filenames(directory):
    filenames = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            filenames.append(file)
    filenames.sort()
    return filenames
data_filenames = get_sorted_filenames(img_dir)
positive_train_files = train_chihuahua
negative_train_files = train_spaniel
unlabelled_train_files = unlabeled_data
# unlabelled_train_files = []
# unlabelled_train_files = unlabelled_train_files = [img_dir+file for file in data_filenames if (file not in positive_train_files) and (file not in negative_train_files)]
# sample_rate = 1
# unlabelled_train_files = random.sample(unlabelled_train_files, int(sample_rate*len(unlabelled_train_files)))
positive_test_files = test_chihuahua
negative_test_files = test_spaniel
uncertain_test_files = []
#nlabelled_train_files = positive_test_files + negative_test_files + uncertain_test_files
#nlabelled_train_files = [file for file in unlabelled_train_files if (file not in positive_train_files) and (file not in negative_train_files)]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--classifier_dir", type=str, required=True)
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config_file", nargs='+', type=str, required=True)
    parser.add_argument("--gin_bindings", default=[], help='Gin bindings to override the values' +
                ' set in the config files (e.g. "DQNAgent.epsilon_train=0.1",' +
                ' "create_atari_environment.game_name="Pong"").')
    args = parser.parse_args()
    load_gin_configs(args.config_file, args.gin_bindings)
    warnings.filterwarnings("ignore", message="Input Tensor 0 did not already require gradients")
    warnings.filterwarnings("ignore", message="Setting forward, backward hooks and attributes")
    warnings.filterwarnings("ignore", message="Lazy modules are a new feature under heavy development")
    set_seed(args.seed)
    base_dir = os.path.join(args.base_dir, str(args.seed))
    classifier = DivDisClassifier(log_dir=os.path.join(base_dir, "logs"))
    classifier.add_data(positive_train_files,
                        negative_train_files,
                        unlabelled_train_files)
    classifier.set_class_weights()
    classifier.train(130) #, progress_bar=True)
    evaluator = DivDisEvaluatorClassifier(
                    classifier,
                    base_dir=base_dir,
                    test_batch_size=64)
    evaluator.add_test_files(positive_test_files, negative_test_files)
    acc_pos, acc_neg, acc, weighted_acc = evaluator.test_classifier()
    print(f"weighted_acc: {weighted_acc}")
    print(f"raw acc:      {acc}")
    print(f"acc_pos:      {acc_pos}")
    print(f"acc_neg:      {acc_neg}")
    evaluator.evaluate_images(250)
    #evaluator.add_true_from_files(positive_test_files)
    #evaluator.add_false_from_files(negative_test_files)
    #evaluator.evaluate(2)
    # print head complexity
    #print(evaluator.get_head_complexity())