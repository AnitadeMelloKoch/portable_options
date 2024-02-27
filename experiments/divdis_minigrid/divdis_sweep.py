import argparse 
from datetime import datetime

from portable.utils.utils import load_gin_configs
from experiments.divdis_minigrid.core.advanced_minigrid_factored_divdis_sweep_experiment import FactoredAdvancedMinigridDivDisSweepExperiment


positive_train_files = [
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_0_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_0_1_termination_positive.npy",
    ]
negative_train_files = [
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_0_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_0_1_termination_negative.npy",
    ]
unlabelled_train_files = [
    # "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getbluekey_doorblue_1_1_termination_positive.npy",
    # "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getbluekey_doorblue_1_1_termination_negative.npy",
    # "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getgreenkey_doorgreen_1_1_termination_negative.npy",
    # "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getgreenkey_doorgreen_1_1_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_2_1_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_1_1_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_2_1_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_1_1_termination_negative.npy",
                          ]

positive_test_files = [
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_5_1_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_6_1_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_7_1_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_8_1_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_9_1_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_10_1_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_11_1_termination_positive.npy",

    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_3_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_4_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_5_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_6_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_7_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_8_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_9_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_10_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_11_termination_positive.npy",
                      ]
negative_test_files = [

    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_5_1_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_6_1_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_7_1_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_8_1_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_9_1_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_10_1_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_11_1_termination_negative.npy",

    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_3_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_4_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_5_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_6_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_7_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_8_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_9_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_10_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_11_termination_negative.npy",

                      ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--config_file", nargs='+', type=str, required=True)
    parser.add_argument("--gin_bindings", default=[], help='Gin bindings to override the values' + 
            ' set in the config files (e.g. "DQNAgent.epsilon_train=0.1",' +
            ' "create_atari_environment.game_name="Pong"").')
    
    args = parser.parse_args()
    load_gin_configs(args.config_file, args.gin_bindings)

    experiment = FactoredAdvancedMinigridDivDisSweepExperiment(base_dir=args.base_dir,
                                                               train_positive_files=positive_train_files,
                                                               train_negative_files=negative_train_files,
                                                               unlabelled_files=unlabelled_train_files,
                                                               test_positive_files=positive_test_files,
                                                               test_negative_files=negative_test_files)

    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{formatted_now}] Sweeping class div weight...")
    experiment.sweep_class_div_weight(0.001,
                                      0.03,
                                      15,
                                      10)

    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{formatted_now}] Sweeping epochs...")
    experiment.sweep_epochs(100, 
                            1000, 
                            100,
                            10)

    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{formatted_now}] Sweeping ensemble size...")
    experiment.sweep_ensemble_size(1, 
                                   25,
                                   10)

    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{formatted_now}] Sweeping div batch size...")
    experiment.sweep_div_batch_size(16,
                                    400,
                                    16,
                                    10)

    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{formatted_now}] Sweeping learning rate...")
    experiment.sweep_lr(0.0001,
                        0.015,
                        15,
                        10)


    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{formatted_now}] Sweeping div overlap...")
    experiment.sweep_div_overlap(0,
                                 1,
                                 5,
                                 10)
    

    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{formatted_now}] Sweeping div variety...")
    
    # Sweep variety
    seed_var = [[0],[1,2],[1,2,3,4]]
    color_var = [['red'],['blue','yellow'],['blue','yellow','green','purple','grey']]
    variety_names = ['low','medium','high']
    rs_var = [False, True]
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
                            file_name = f"resources/factored_minigrid_images/adv_doorkey_8x8_v2_get{c}key_door{c}_{s}_1_termination_positive.npy"
                        else:
                            file_name = f"resources/factored_minigrid_images/adv_doorkey_8x8_v2_get{c}key_door{c}_{s}_termination_positive.npy"
                        combination_files.append(file_name)
                all_combination_files.append(combination_files)
                variety_combinations.append(f'{variety_names[seed_idx]},{variety_names[color_idx]},{"included" if rs_var[rs_idx] else "none"}')
        
    experiment.sweep_div_variety(variety_combinations, all_combination_files, 10)


