import argparse 
from datetime import datetime

import numpy as np

from portable.utils.utils import load_gin_configs
from experiments.divdis_minigrid.core.advanced_minigrid_divdis_sweep_experiment import AdvancedMinigridDivDisSweepExperiment

color = 'red'
task = f'get{color}key'
#task = f'open{color}door'
init_term = 'termination'

#RANDOM_TRAIN = True
#RANDOM_UNLABELLED = True

base_img_dir = 'resources/minigrid_images/'
positive_train_files = [f"{base_img_dir}adv_doorkey_16x16_v2_{task}_door{color}_0_1_{init_term}_positive.npy"]
negative_train_files = [f"{base_img_dir}adv_doorkey_16x16_v2_{task}_door{color}_0_1_{init_term}_negative.npy"]
unlabelled_train_files = [f"{base_img_dir}adv_doorkey_16x16_v2_{task}_door{color}_{s}_1_{init_term}_{pos_neg}.npy" for s in [1,2,3,4] for pos_neg in ['positive', 'negative']]
positive_test_files = [f"{base_img_dir}adv_doorkey_16x16_v2_{task}_door{color}_{s}_1_{init_term}_positive.npy" for s in [5,6,7,8,9,10,11]]
negative_test_files = [f"{base_img_dir}adv_doorkey_16x16_v2_{task}_door{color}_{s}_1_{init_term}_negative.npy" for s in [5,6,7,8,9,10,11]]


def formatted_time():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--config_file", nargs='+', type=str, required=True)
    parser.add_argument("--gin_bindings", default=[], help='Gin bindings to override the values' + 
            ' set in the config files (e.g. "DQNAgent.epsilon_train=0.1",' +
            ' "create_atari_environment.game_name="Pong"").')
    
    args = parser.parse_args()
    load_gin_configs(args.config_file, args.gin_bindings)

    experiment = AdvancedMinigridDivDisSweepExperiment(base_dir=args.base_dir,
                                                       train_positive_files=positive_train_files,
                                                       train_negative_files=negative_train_files,
                                                       unlabelled_files=unlabelled_train_files,
                                                       test_positive_files=positive_test_files,
                                                       test_negative_files=negative_test_files)
    
    NUM_SEEDS = 5


    
    print(f"[{formatted_time()}] Sweeping learning rate...")
    experiment.sweep_lr(-4, # 0.0001
                        -3,
                        10,
                        NUM_SEEDS)
    


    print(f"[{formatted_time()}] Sweeping class div weight...")
    experiment.sweep_class_div_weight(-5, # 0.00001
                                      -2,
                                      15,
                                      NUM_SEEDS)

    print(f"[{formatted_time()}] Sweeping L2 reg weight...")
    experiment.sweep_l2_reg_weight(-4, # 0.0001
                                   -2,
                                   10,
                                   NUM_SEEDS)



    print(f"[{formatted_time()}] Sweeping ensemble size...")
    experiment.sweep_ensemble_size(1, 
                                   8,
                                   1,
                                   NUM_SEEDS)

    
    print(f"[{formatted_time()}] Sweeping epochs...")
    experiment.sweep_epochs(5, 
                            100, 
                            10,
                            NUM_SEEDS)

    print(f"[{formatted_time()}] Sweeping div batch size...")
    experiment.sweep_div_batch_size(16,
                                    400,
                                    16,
                                    NUM_SEEDS)

    #print(f"[{formatted_time()}] Now running grid search...")
    #experiment.grid_search(lr_range=np.logspace(-4, -3, 10),
    #                        div_weight_range=np.logspace(-5, -2, 15),
    #                        l2_reg_range=np.logspace(-4, -2, 10),
    #                        head_num_range=range(1, 8),
    #                        epochs_range=range(5, 55, 10),
    #                        num_seeds=NUM_SEEDS)



    print(f"[{formatted_time()}] Sweeping div overlap...")
    experiment.sweep_div_overlap(0,
                                 1,
                                 5,
                                 NUM_SEEDS)

    
    print(f"[{formatted_time()}] Sweeping div variety...")
    # Sweep variety
    seed_var = [[0],[1,2],[1,2,3,4]]
    color_var = [['red'],['blue','yellow'],['blue','yellow','green','purple','grey']]
    variety_names = ['low','medium','high']
    rs_var = [True]
    variety_combinations = []
    all_combination_files = []
    
    for seed_idx in range(len(seed_var)):
        for color_idx in range(len(color_var)):
            for rs_idx in range(len(rs_var)):
                # 3*3*2 = 18 combinations
                combination_files = []
                for s in seed_var[seed_idx]:
                    for c in color_var[color_idx]:
                        if rs_var[rs_idx]:
                            file_name = f"resources/minigrid_images/adv_doorkey_16x16_v2_{task}_door{c}_{s}_1_termination_positive.npy"
                        else:
                            file_name = f"resources/minigrid_images/adv_doorkey_16x16_v2_{task}_door{c}_{s}_termination_positive.npy"
                        combination_files.append(file_name)
                all_combination_files.append(combination_files)
                variety_combinations.append(f'{variety_names[seed_idx]},{variety_names[color_idx]},{"included" if rs_var[rs_idx] else "none"}')
        
    experiment.sweep_div_variety(variety_combinations, all_combination_files, NUM_SEEDS)


