from experiments.divdis_minigrid.core.advanced_minigrid_factored_divdis_experiment import FactoredAdvancedMinigridDivDisExperiment
import argparse
from portable.utils.utils import load_gin_configs
import torch 
from experiments.minigrid.utils import factored_environment_builder
from experiments.minigrid.advanced_doorkey.core.policy_train_wrapper import AdvancedDoorKeyPolicyTrainWrapper
from experiments.minigrid.advanced_doorkey.advanced_minigrid_option_resources import *
import random

def make_random_getkey_env(train_colour, seed):
    colours = ["red", "green", "blue", "purple", "yellow", "grey"]
    possible_key_colours = list(filter(lambda c: c!= train_colour, colours))
    
    door_colour = random.choice(possible_key_colours)
    possible_key_colours = list(filter(lambda c: c!= door_colour, possible_key_colours))
    random.shuffle(possible_key_colours)
    key_cols = [train_colour] + possible_key_colours
    
    return AdvancedDoorKeyPolicyTrainWrapper(
        factored_environment_builder(
            'AdvancedDoorKey-8x8-v0',
            seed=seed,
        ),
        door_colour=door_colour,
        key_colours=key_cols,
        time_limit=100,
        image_input=False
    )

positive_train_files = [
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_0_termination_positive.npy",
    ]
negative_train_files = [
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_0_termination_negative.npy",
    ]
unlabelled_train_files = [
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getbluekey_doorblue_1_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getbluekey_doorblue_1_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_2_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_1_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_2_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_1_termination_negative.npy",
                          ]

positive_test_files = [
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_3_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_4_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_5_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_6_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_7_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_8_termination_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_9_termination_positive.npy",
                      ]
negative_test_files = [
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_3_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_4_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_5_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_6_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_7_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_8_termination_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_getredkey_doorred_9_termination_negative.npy",
                      ]




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
    
    def policy_phi(x):
        x = x/torch.tensor([7,7,1,1,5,7,7,5,7,7,5,7,7,5,7,7,5,7,7,5,7,7,4,7,7,7])

        return x
    
    experiment = FactoredAdvancedMinigridDivDisExperiment(base_dir=args.base_dir,
                                                          seed=args.seed,
                                                          policy_phi=policy_phi)
    
    
    
    experiment.add_datafiles(positive_train_files,
                             negative_train_files,
                             unlabelled_train_files)
    
    experiment.train_termination(20,
                                 positive_test_files,
                                 negative_test_files)
    
    total_acc, accuracy = experiment.test_terminations(positive_test_files,
                                            negative_test_files)
    
    print(accuracy)
    
    for idx, acc in enumerate(accuracy):
        print(idx, acc)
        if acc > 0.55:
            for seed in range(3,10):
                envs = [AdvancedDoorKeyPolicyTrainWrapper(
                                factored_environment_builder(
                                    'AdvancedDoorKey-8x8-v0',
                                    seed=seed
                                ),
                                door_colour="red",
                                time_limit=100,
                                image_input=False
                            ),
                        make_random_getkey_env("red", seed),
                        make_random_getkey_env("red", seed),
                        make_random_getkey_env("red", seed),
                        ]
                experiment.train_policy(max_steps=3e6,
                                        min_performance=0.9,
                                        envs=envs,
                                        idx=idx,
                                        seed=seed,
                                        perfect_term=check_got_redkey)
                env = AdvancedDoorKeyPolicyTrainWrapper(
                    factored_environment_builder(
                        'AdvancedDoorKey-8x8-v0',
                        seed=seed
                    ),
                    door_colour="red",
                    time_limit=100,
                    image_input=False,
                    check_option_complete=check_got_redkey
                )
                
                experiment.evaluate_policy(envs=[env],
                                        trials_per_env=10,
                                        idx=idx,
                                        seed=seed)
            










