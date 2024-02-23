from experiments.divdis_minigrid.core.advanced_minigrid_factored_divdis_sweep_experiment import FactoredAdvancedMinigridDivDisSweepExperiment
import argparse 
from portable.utils.utils import load_gin_configs

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
    '''experiment.sweep_class_div_weight(0.005,
                                      0.015,
                                      10,
                                      5)
    
    experiment.sweep_epochs(50, 
                            300, 
                            50,
                            5)
    
    experiment.sweep_ensemble_size(1, 
                                   20,
                                   5)
    
    experiment.sweep_div_batch_size(16,
                                    400,
                                    16,
                                    5)
    
    experiment.sweep_lr(0.001,
                        0.015,
                        10,
                        5)

    experiment.sweep_div_overlap(0,
                                 1,
                                 4,
                                 5)

    # Sweep variety
    seed_var = [[0],[1,2],[1,2,3,4]]
    color_var = [['red'],['blue','yellow'],['blue','yellow','green','purple','grey']]
    rs_var = [False, True]
    variety_combinations = []
    all_combination_files = []
    
    for seed in seed_var:
        for color in color_var:
            for rs in rs_var:
                # 3*3*2 = 18 combinations
                combination_files = []
                for s in seed:
                    for c in color:
                        if rs:
                            file_name = f"resources/factored_minigrid_images/adv_doorkey_8x8_v2_open{c}door_door{c}_{s}_1_initiation_positive.npy"
                        else:
                            file_name = f"resources/factored_minigrid_images/adv_doorkey_8x8_v2_open{c}door_door{c}_{s}_initiation_positive.npy"
                        combination_files.append(file_name)
                variety_combinations.append((seed, color, rs))
                all_combination_files.append(combination_files)

    experiment.sweep_div_variety(variety_combinations, all_combination_files, 5)'''
    experiment.sweep_epochs(50, 
                            600, 
                            100,
                            10)
