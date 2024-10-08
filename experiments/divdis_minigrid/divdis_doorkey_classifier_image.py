import argparse
import random
import time

import numpy as np
import torch
from tqdm import tqdm
#import torch_tensorrt

from portable.utils.utils import load_gin_configs
from experiments.divdis_minigrid.core.advanced_minigrid_divdis_classifier_experiment import \
    AdvancedMinigridDivDisClassifierExperiment

color = 'grey'
#task = f'open{color}door'
task = f'get{color}key'
init_term = 'termination'
RANDOM_TRAIN = True
RANDOM_UNLABELLED = True


base_img_dir = 'resources/minigrid_images/'
positive_train_files = [f"{base_img_dir}adv_doorkey_8x8_v2_{task}_door{color}_0_{init_term}_positive.npy"]
negative_train_files = [f"{base_img_dir}adv_doorkey_8x8_v2_{task}_door{color}_0_{init_term}_negative.npy"]
unlabelled_train_files = [f"{base_img_dir}adv_doorkey_8x8_v2_{task}_door{color}_{s}_{init_term}_{pos_neg}.npy" for s in [1,2] for pos_neg in ['positive', 'negative']]
positive_test_files = [f"{base_img_dir}adv_doorkey_8x8_v2_{task}_door{color}_{s}_{init_term}_positive.npy" for s in [3,4,5,6,7,8,9,10]]
negative_test_files = [f"{base_img_dir}adv_doorkey_8x8_v2_{task}_door{color}_{s}_{init_term}_negative.npy" for s in [3,4,5,6,7,8,9,10]]

if RANDOM_TRAIN:
    positive_train_files += [f"{base_img_dir}adv_doorkey_8x8_v2_{task}_door{color}_0_1_{init_term}_positive.npy"]
    negative_train_files += [f"{base_img_dir}adv_doorkey_8x8_v2_{task}_door{color}_0_1_{init_term}_negative.npy"]
if RANDOM_UNLABELLED:
    unlabelled_train_files += [f"{base_img_dir}adv_doorkey_8x8_v2_{task}_door{color}_{s}_1_{init_term}_{pos_neg}.npy" for s in [1,2] for pos_neg in ['positive', 'negative']]
positive_test_files += [f"{base_img_dir}adv_doorkey_8x8_v2_{task}_door{color}_{s}_1_{init_term}_positive.npy" for s in [3,4,5,6,7,8,9,10]]
negative_test_files += [f"{base_img_dir}adv_doorkey_8x8_v2_{task}_door{color}_{s}_1_{init_term}_negative.npy" for s in [3,4,5,6,7,8,9,10]]
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--config_file", nargs='+', type=str, required=True)
    parser.add_argument("--gin_bindings", default=[], help='Gin bindings to override the values' + 
        ' set in the config files (e.g. "DQNAgent.epsilon_train=0.1",' +
        ' "create_atari_environment.game_name="Pong"").')

    args = parser.parse_args()

    load_gin_configs(args.config_file, args.gin_bindings)

    #torch.set_float32_matmul_precision('high')
    #torch.backends.cuda.matmul.allow_tf32 = True
    #torch.set_float32_matmul_precision('medium')
    #torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    #print(torch._dynamo.list_backends())
    seeds = [20*i + args.seed for i in range(args.n)]

    best_total_acc = []
    best_weighted_acc = []
    avg_weighted_acc = []
    train_time = []
    test_time = []

    for i in tqdm(range(args.n)):
        cur_seed = seeds[i]
            
        t0 = time.time()
        experiment = AdvancedMinigridDivDisClassifierExperiment(
                            base_dir=args.base_dir,
                            seed=cur_seed)

        experiment.add_datafiles(positive_train_files,
                    negative_train_files,
                    unlabelled_train_files)

        experiment.train_classifier()
        
        t1 = time.time()
        train_time.append(t1-t0)
        t2 = time.time()
        accuracy = experiment.test_classifier(positive_test_files,
                            negative_test_files)
        test_time.append(time.time()-t2)
        
        print(f"Total Accuracy: {np.round(accuracy[0], 4)}")
        print(f"Weighted Accuracy: {np.round(accuracy[1], 4)}")
        best_head = np.argmax(accuracy[1])
        best_weighted_acc.append(accuracy[1][best_head])
        best_total_acc.append(accuracy[0][best_head])
        avg_weighted_acc.append(np.mean(accuracy[1]))

    print(f"Best Total Accuracy: {np.mean(best_total_acc):.2f}, {np.std(best_total_acc):.2f}")
    print(f"Best Weighted Accuracy: {np.mean(best_weighted_acc):.2f}, {np.std(best_weighted_acc):.2f}")
    print(f"Avg Weighted Accuracy: {np.mean(avg_weighted_acc):.2f}, {np.std(avg_weighted_acc):.2f}")
    print(f"Train Time: {np.mean(train_time):.2f}, {np.std(train_time):.2f}")
    print(f"Test Time: {np.mean(test_time):.2f}, {np.std(test_time):.2f}")

    