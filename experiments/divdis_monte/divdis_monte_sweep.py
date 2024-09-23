import argparse 
from datetime import datetime
import warnings

import numpy as np

from portable.utils.utils import load_gin_configs
from experiments.divdis_monte.core.divdis_monte_sweep_experiment import MonteDivDisSweepExperiment


img_dir = "resources/monte_images/"


positive_train_files = [img_dir+"climb_down_ladder_room1_termination_positive.npy",
                        #img_dir+"climb_down_ladder_room10_termination_positive.npy",
                        ]

negative_train_files = [img_dir+"climb_down_ladder_room1_termination_negative.npy",
                        img_dir+"climb_down_ladder_room1_1_termination_negative.npy",
                        img_dir+"screen_death_1.npy",
                        img_dir+"screen_death_2.npy",
                        img_dir+"screen_death_3.npy",
                        img_dir+"screen_death_4.npy",
                        #img_dir+"climb_down_ladder_room10_termination_negative.npy",
                        #img_dir+"climb_down_ladder_room10_uncertain.npy"
                        ]
unlabelled_train_files = [
    # 0
    img_dir + "climb_up_ladder_room0_termination_positive.npy",
     img_dir + "climb_up_ladder_room0_termination_negative.npy",
     img_dir + "climb_up_ladder_room0_uncertain.npy",
     img_dir+"climb_down_ladder_room0_termination_negative.npy",
     img_dir+"climb_down_ladder_room0_uncertain.npy",
    # 4
    img_dir + "climb_up_ladder_room4_termination_negative.npy",
     img_dir + "climb_up_ladder_room4_uncertain.npy",
     img_dir+"climb_down_ladder_room4_termination_negative.npy",
     img_dir+"climb_down_ladder_room4_uncertain.npy",
     img_dir + "move_right_enemy_room4_termination_positive.npy",
     img_dir + "move_right_enemy_room4_termination_negative.npy",
     img_dir + "move_left_enemy_room4_termination_positive.npy",
     img_dir + "move_left_enemy_room4_termination_negative.npy",
    # 3
    img_dir + "climb_up_ladder_room3_termination_positive.npy",
     img_dir + "climb_up_ladder_room3_termination_negative.npy",
     img_dir + "climb_up_ladder_room3_uncertain.npy",
     img_dir+"climb_down_ladder_room3_termination_negative.npy",
     img_dir+"climb_down_ladder_room3_uncertain.npy",
     img_dir + "move_right_enemy_room3_termination_positive.npy",
     img_dir + "move_right_enemy_room3_termination_negative.npy",
     img_dir + "move_left_enemy_room3_termination_positive.npy",
     img_dir + "move_left_enemy_room3_termination_negative.npy",
    # 9
    img_dir + "climb_up_ladder_room9_termination_negative.npy",
     img_dir + "climb_up_ladder_room9_uncertain.npy",
     img_dir+"climb_down_ladder_room9_termination_positive.npy",
     img_dir+"climb_down_ladder_room9_termination_negative.npy",
     img_dir+"climb_down_ladder_room9_uncertain.npy",
     img_dir + "move_right_enemy_room9left_termination_positive.npy",
     img_dir + "move_right_enemy_room9left_termination_negative.npy",
     img_dir + "move_right_enemy_room9right_termination_positive.npy",
     img_dir + "move_right_enemy_room9right_termination_negative.npy",
     img_dir + "move_left_enemy_room9left_termination_positive.npy",
     img_dir + "move_left_enemy_room9left_termination_negative.npy",
     img_dir + "move_left_enemy_room9right_termination_positive.npy",
     img_dir + "move_left_enemy_room9right_termination_negative.npy",
    # 8
    img_dir + "room8_walk_around.npy",
    # 10
    img_dir + "climb_up_ladder_room10_termination_negative.npy",
     img_dir + "climb_up_ladder_room10_uncertain.npy",
     img_dir+"climb_down_ladder_room10_termination_negative.npy",
     img_dir+"climb_down_ladder_room10_termination_positive.npy",
     img_dir+"climb_down_ladder_room10_uncertain.npy",
    # 11
    img_dir + "climb_up_ladder_room11_termination_negative.npy",
     img_dir + "climb_up_ladder_room11_uncertain.npy",
     img_dir+"climb_down_ladder_room11_termination_negative.npy",
     img_dir+"climb_down_ladder_room11_uncertain.npy",
     img_dir + "move_right_enemy_room11left_termination_positive.npy",
     img_dir + "move_right_enemy_room11left_termination_negative.npy",
     img_dir + "move_right_enemy_room11right_termination_positive.npy",
     img_dir + "move_right_enemy_room11right_termination_negative.npy",
     img_dir + "move_left_enemy_room11left_termination_positive.npy",
     img_dir + "move_left_enemy_room11left_termination_negative.npy",
     img_dir + "move_left_enemy_room11right_termination_positive.npy",
     img_dir + "move_left_enemy_room11right_termination_negative.npy",
    # 5
    img_dir + "climb_up_ladder_room5_termination_positive.npy",
     #img_dir + "climb_up_ladder_room5_termination_negative.npy",
     img_dir + "climb_up_ladder_room5_uncertain.npy",
     img_dir+"climb_down_ladder_room5_termination_negative.npy",
     img_dir+"climb_down_ladder_room5_uncertain.npy",
     img_dir + "move_right_enemy_room5_termination_positive.npy",
     img_dir + "move_right_enemy_room5_termination_negative.npy",
     img_dir + "move_left_enemy_room5_termination_positive.npy",
     img_dir + "move_left_enemy_room5_termination_negative.npy",
    # 13
    img_dir + "climb_up_ladder_room13_termination_negative.npy",
     img_dir + "climb_up_ladder_room13_uncertain.npy",
     img_dir+"climb_down_ladder_room13_termination_negative.npy",
     img_dir+"climb_down_ladder_room13_uncertain.npy",
     img_dir + "move_right_enemy_room13_termination_positive.npy",
     img_dir + "move_right_enemy_room13_termination_negative.npy",
     img_dir + "move_left_enemy_room13_termination_positive.npy",
     img_dir + "move_left_enemy_room13_termination_negative.npy",
    # 7
    img_dir + "climb_up_ladder_room7_termination_positive.npy",
     img_dir + "climb_up_ladder_room7_termination_negative.npy",
     img_dir + "climb_up_ladder_room7_uncertain.npy",
     img_dir+"climb_down_ladder_room7_termination_negative.npy",
     img_dir+"climb_down_ladder_room7_uncertain.npy",
    # 6
    img_dir + "climb_up_ladder_room6_termination_negative.npy",
     img_dir + "climb_up_ladder_room6_uncertain.npy",
     img_dir+"climb_down_ladder_room6_termination_positive.npy",
     img_dir+"climb_down_ladder_room6_termination_negative.npy",
     img_dir+"climb_down_ladder_room6_uncertain.npy",
    # 2
    img_dir + "climb_up_ladder_room2_termination_positive.npy",
     img_dir + "climb_up_ladder_room2_termination_negative.npy",
     img_dir + "climb_up_ladder_room2_uncertain.npy",
     img_dir+"climb_down_ladder_room2_termination_negative.npy",
     img_dir+"climb_down_ladder_room2_uncertain.npy",
     img_dir + "move_right_enemy_room2_termination_positive.npy",
     img_dir + "move_right_enemy_room2_termination_negative.npy",
     img_dir + "move_left_enemy_room2_termination_positive.npy",
     img_dir + "move_left_enemy_room2_termination_negative.npy",
    # 14
    img_dir + "climb_up_ladder_room14_termination_positive.npy",
     img_dir + "climb_up_ladder_room14_termination_negative.npy",
     img_dir + "climb_up_ladder_room14_uncertain.npy",
     img_dir+"climb_down_ladder_room14_termination_negative.npy",
     img_dir+"climb_down_ladder_room14_uncertain.npy",
    # 22
    img_dir + "climb_up_ladder_room22_termination_negative.npy",
     img_dir + "climb_up_ladder_room22_uncertain.npy",
     img_dir+"climb_down_ladder_room22_termination_negative.npy",
     img_dir+"climb_down_ladder_room22_termination_positive.npy",
     img_dir+"climb_down_ladder_room22_uncertain.npy",
     img_dir + "move_right_enemy_room22_termination_positive.npy",
     img_dir + "move_right_enemy_room22_termination_negative.npy",
     img_dir + "move_left_enemy_room22_termination_positive.npy",
     img_dir + "move_left_enemy_room22_termination_negative.npy",
    # 21
    img_dir + "climb_up_ladder_room21_termination_negative.npy",
     img_dir + "climb_up_ladder_room21_uncertain.npy",
     img_dir+"climb_down_ladder_room21_termination_positive.npy",
     img_dir+"climb_down_ladder_room21_termination_negative.npy",
     img_dir+"climb_down_ladder_room21_uncertain.npy",
     img_dir + "move_right_enemy_room21_termination_positive.npy",
     img_dir + "move_right_enemy_room21_termination_negative.npy",
     img_dir + "move_left_enemy_room21_termination_positive.npy",
     img_dir + "move_left_enemy_room21_termination_negative.npy",
    # 19
    img_dir + "climb_up_ladder_room19_termination_negative.npy",
     img_dir + "climb_up_ladder_room19_uncertain.npy",
     img_dir+"climb_down_ladder_room19_termination_positive.npy",
     img_dir+"climb_down_ladder_room19_termination_negative.npy",
     img_dir+"climb_down_ladder_room19_uncertain.npy",
    # 18
    img_dir + "room18_walk_around.npy",
     img_dir + "move_left_enemy_room18_termination_positive.npy",
     img_dir + "move_left_enemy_room18_termination_negative.npy"
]


positive_test_files = [img_dir+"climb_down_ladder_room1_termination_positive.npy",
                       img_dir+"climb_down_ladder_room6_termination_positive.npy",
                       img_dir+"climb_down_ladder_room9_termination_positive.npy",
                       img_dir+"climb_down_ladder_room10_termination_positive.npy",
                       img_dir+"climb_down_ladder_room19_termination_positive.npy",
                       img_dir+"climb_down_ladder_room21_termination_positive.npy",
                       img_dir+"climb_down_ladder_room22_termination_positive.npy",
                       
                       img_dir+"climb_down_ladder_room6_1_termination_positive.npy",
                       img_dir+"climb_down_ladder_room9_1_termination_positive.npy",
                       img_dir+"climb_down_ladder_room10_1_termination_positive.npy",
                       img_dir+"climb_down_ladder_room19_1_termination_positive.npy",
                       img_dir+"climb_down_ladder_room21_1_termination_positive.npy",
                       img_dir+"climb_down_ladder_room22_1_termination_positive.npy",
                       ]
negative_test_files = [img_dir+"screen_death_1.npy",
                       img_dir+"screen_death_2.npy",
                       img_dir+"screen_death_3.npy",
                       img_dir+"screen_death_4.npy",
                       img_dir+"climb_down_ladder_room1_termination_negative.npy",
                       img_dir+"climb_down_ladder_room2_termination_negative.npy",
                       img_dir+"climb_down_ladder_room3_termination_negative.npy",
                       img_dir+"climb_down_ladder_room4_termination_negative.npy",
                       img_dir+"climb_down_ladder_room5_termination_negative.npy",
                       img_dir+"climb_down_ladder_room6_termination_negative.npy",
                       img_dir+"climb_down_ladder_room7_termination_negative.npy",
                       img_dir+"climb_down_ladder_room9_termination_negative.npy",
                       img_dir+"climb_down_ladder_room10_termination_negative.npy",
                       img_dir+"climb_down_ladder_room11_termination_negative.npy",
                       img_dir+"climb_down_ladder_room13_termination_negative.npy",
                       img_dir+"climb_down_ladder_room14_termination_negative.npy",
                       img_dir+"climb_down_ladder_room19_termination_negative.npy",
                       img_dir+"climb_down_ladder_room21_termination_negative.npy",
                       img_dir+"climb_down_ladder_room22_termination_negative.npy",
                       
                       img_dir+"climb_down_ladder_room1_1_termination_negative.npy",
                       img_dir+"climb_down_ladder_room2_1_termination_negative.npy",
                       img_dir+"climb_down_ladder_room3_1_termination_negative.npy",
                       img_dir+"climb_down_ladder_room4_1_termination_negative.npy",
                       img_dir+"climb_down_ladder_room5_1_termination_negative.npy",
                       img_dir+"climb_down_ladder_room6_1_termination_negative.npy",
                       img_dir+"climb_down_ladder_room7_1_termination_negative.npy",
                       img_dir+"climb_down_ladder_room9_1_termination_negative.npy",
                       img_dir+"climb_down_ladder_room10_1_termination_negative.npy",
                       img_dir+"climb_down_ladder_room11_1_termination_negative.npy",
                       img_dir+"climb_down_ladder_room13_1_termination_negative.npy",
                       img_dir+"climb_down_ladder_room14_1_termination_negative.npy",
                       img_dir+"climb_down_ladder_room19_1_termination_negative.npy",
                       img_dir+"climb_down_ladder_room21_1_termination_negative.npy",
                       img_dir+"climb_down_ladder_room22_1_termination_negative.npy",        
                       ]
uncertain_test_files = [img_dir+"climb_down_ladder_room0_uncertain.npy",
                        img_dir+"climb_down_ladder_room2_uncertain.npy",
                        img_dir+"climb_down_ladder_room3_uncertain.npy",
                        img_dir+"climb_down_ladder_room4_uncertain.npy",
                        img_dir+"climb_down_ladder_room5_uncertain.npy",
                        img_dir+"climb_down_ladder_room6_uncertain.npy",
                        img_dir+"climb_down_ladder_room7_uncertain.npy",
                        img_dir+"climb_down_ladder_room9_uncertain.npy",
                        img_dir+"climb_down_ladder_room10_uncertain.npy",
                        img_dir+"climb_down_ladder_room11_uncertain.npy",
                        img_dir+"climb_down_ladder_room13_uncertain.npy",
                        img_dir+"climb_down_ladder_room14_uncertain.npy",
                        img_dir+"climb_down_ladder_room19_uncertain.npy",
                        img_dir+"climb_down_ladder_room21_uncertain.npy",
                        img_dir+"climb_down_ladder_room22_uncertain.npy",

                        img_dir+"climb_down_ladder_room3_1_uncertain.npy",
                        img_dir+"climb_down_ladder_room4_1_uncertain.npy",
                        img_dir+"climb_down_ladder_room6_1_uncertain.npy",
                        img_dir+"climb_down_ladder_room9_1_uncertain.npy",
                        img_dir+"climb_down_ladder_room10_1_uncertain.npy",
                        img_dir+"climb_down_ladder_room11_1_uncertain.npy",
                        img_dir+"climb_down_ladder_room13_1_uncertain.npy",
                        img_dir+"climb_down_ladder_room19_1_uncertain.npy",
                        img_dir+"climb_down_ladder_room21_1_uncertain.npy",
                        img_dir+"climb_down_ladder_room22_1_uncertain.npy",
                        ]

def formatted_time():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

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

    warnings.filterwarnings("ignore", category=UserWarning, module='torch.nn.modules.lazy')


    experiment = MonteDivDisSweepExperiment(base_dir=args.base_dir,
                                            train_positive_files=positive_train_files,
                                            train_negative_files=negative_train_files,
                                            unlabelled_files=unlabelled_train_files,
                                            test_positive_files=positive_test_files,
                                            test_negative_files=negative_test_files,
                                            seed=args.seed)
    
    NUM_SEEDS = 5

    print(f"[{formatted_time()}] Now running grid search...")
    experiment.grid_search(lr_range=np.logspace(-4, -3, 3),
                            div_weight_range=np.logspace(-4, -2, 4),
                            l2_reg_range=np.logspace(-4, -2, 3),
                            head_num_range=[4,6,8],
                            epochs_range=[30], #[30,70,150,300]
                            num_seeds=NUM_SEEDS)

    
    """print(f"[{formatted_time()}] Sweeping learning rate...")
    experiment.sweep_lr(-5, # 0.00001
                        -3,
                        8,
                        NUM_SEEDS)


    print(f"[{formatted_time()}] Sweeping class div weight...")
    experiment.sweep_class_div_weight(-5, # 0.00001
                                      -3,
                                      8,
                                      NUM_SEEDS)

    print(f"[{formatted_time()}] Sweeping L2 reg weight...")
    experiment.sweep_l2_reg_weight(-4, # 0.0001
                                   -2,
                                   8,
                                   NUM_SEEDS)


    print(f"[{formatted_time()}] Sweeping ensemble size...")
    experiment.sweep_ensemble_size(1, 
                                   8,
                                   1,
                                   NUM_SEEDS,
                                   )

    
    print(f"[{formatted_time()}] Sweeping epochs...")
    experiment.sweep_epochs(5, 
                            100, 
                            10,
                            NUM_SEEDS,
                            [30,60,100,130,160,200,250,350,500]) # when a list is provided, use this


    print(f"[{formatted_time()}] Sweeping div batch size...")
    experiment.sweep_div_batch_size(16,
                                    400,
                                    16,
                                    NUM_SEEDS)"""

    
    
    
    
    
    



    #print(f"[{formatted_time()}] Sweeping div overlap...")
    #experiment.sweep_div_overlap(0,
    #                             1,
    #                             5,
    #                             NUM_SEEDS)
    #
    #
    #print(f"[{formatted_time()}] Sweeping div variety...")
    ## Sweep variety
    #seed_var = [[0],[1,2],[1,2,3,4]]
    #color_var = [['red'],['blue','yellow'],['blue','yellow','green','purple','grey']]
    #variety_names = ['low','medium','high']
    #rs_var = [True]
    #variety_combinations = []
    #all_combination_files = []
    #
    #for seed_idx in range(len(seed_var)):
    #    for color_idx in range(len(color_var)):
    #        for rs_idx in range(len(rs_var)):
    #            # 3*3*2 = 18 combinations
    #            combination_files = []
    #            for s in seed_var[seed_idx]:
    #                for c in color_var[color_idx]:
    #                    if rs_var[rs_idx]:
    #                        file_name = f"resources/minigrid_images/adv_doorkey_16x16_v2_{task}_door{c}_{s}_1_termination_positive.npy"
    #                    else:
    #                        file_name = f"resources/minigrid_images/adv_doorkey_16x16_v2_{task}_door{c}_{s}_termination_positive.npy"
    #                    combination_files.append(file_name)
    #            all_combination_files.append(combination_files)
    #            variety_combinations.append(f'{variety_names[seed_idx]},{variety_names[color_idx]},{"included" if rs_var[rs_idx] else "none"}')
    #    
    #experiment.sweep_div_variety(variety_combinations, all_combination_files, NUM_SEEDS)


