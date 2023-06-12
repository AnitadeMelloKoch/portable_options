import os 
import argparse 
import numpy as np
from portable.utils import load_gin_configs
from portable.option import Option
import torch
from experiments.factored_minigrid.utils import environment_builder

from evaluation.evaluators import AttentionEvaluatorPolicy, AttentionEvaluatorClassifier

stack_size=6

initiation_positive_files = [
    'resources/minigrid_images/doorkey_getkey_1_initiation_positive.npy'
]

initiation_negative_files = [
    'resources/minigrid_images/doorkey_getkey_1_initiation_negative.npy'
]

termination_positive_files = [
    'resources/minigrid_images/doorkey_getkey_1_termination_positive.npy'
]

termination_negative_files = [
    'resources/minigrid_images/doorkey_getkey_1_termination_negative.npy'
]

def policy_phi(x):
    if np.max(x) > 1:
        x = x/255.0
    x = x.astype(np.float32)
    return x

def get_latent_state_function(env, info):
    return

def markov_option_builder(todo_check_args):
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--load_dir", type=str, required=True)
    parser.add_argument("--plot_dir", type=str, required=True)
    parser.add_argument("--policy", action="store_true")
    parser.add_argument("--term", action="store_true")
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--config_file", nargs='+', type=str, required=True)
    
    args = parser.parse_args()
    
    device = torch.device("cuda")
    
    load_gin_configs(args.config_file, [])
    
    option = Option(device,
                    markov_option_builder=markov_option_builder,
                    get_latent_state=get_latent_state_function,
                    policy_phi=policy_phi,
                    initiation_vote_function="weighted_vote_low",
                    termination_vote_function="weighted_vote_low")
    
    option.load(args.load_dir)
    
    if not (args.policy or args.init or args.term):
        raise Exception('Must test an element')
    
    if args.policy:
        evaluator = AttentionEvaluatorPolicy(option.policy.value_ensemble,
                                             os.path.join(args.plot_dir, 'policy'),
                                             stack_size)
        env = environment_builder('FactoredMiniGrid-DoorKey-8x8-v0',
                        seed=12)
        state, info = env.reset()
        evaluator.evaluate(env, state)
        
    if args.init or args.term:
        if args.init:
            evaluator = AttentionEvaluatorClassifier(option.initiation.classifier,
                                                    os.path.join(args.plot_dir, 'initiation'),
                                                    stack_size)
            positive_files = initiation_positive_files
            negative_files = initiation_negative_files
            evaluator.add_true_from_files(positive_files)
            evaluator.add_false_from_files(negative_files)
            evaluator.evaluate(20)
        if args.term:
            evaluator = AttentionEvaluatorClassifier(option.termination.classifier,
                                                    os.path.join(args.plot_dir, 'termination'),
                                                    stack_size)
            positive_files = termination_positive_files
            negative_files = termination_negative_files
        
            evaluator.add_true_from_files(positive_files)
            evaluator.add_false_from_files(negative_files)
            evaluator.evaluate(20)
        

