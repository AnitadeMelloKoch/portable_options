from experiments.divdis_minigrid.core.advanced_minigrid_factored_divdis_experiment import FactoredAdvancedMinigridDivDisExperiment
import argparse
from portable.utils.utils import load_gin_configs
import torch 
import numpy as np
from experiments.minigrid.utils import factored_environment_builder
from experiments.minigrid.advanced_doorkey.core.policy_train_wrapper import AdvancedDoorKeyPolicyTrainWrapper
from experiments.minigrid.advanced_doorkey.advanced_minigrid_option_resources import *
from collections import deque

positive_train_files = ["resources/minigrid_factored/adv_doorkey_8x8_openreddoor_doorred_2_termination_positive.npy"]
negative_train_files = ["resources/minigrid_factored/adv_doorkey_8x8_openreddoor_doorred_2_termination_negative.npy"]
unlabelled_train_files = ["resources/minigrid_factored/adv_doorkey_8x8_openbluedoor_doorblue_1_termination_positive.npy",
                          "resources/minigrid_factored/adv_doorkey_8x8_openbluedoor_doorblue_0_termination_negative.npy",
                          "resources/minigrid_factored/adv_doorkey_8x8_openyellowdoor_dooryellow_2_termination_positive.npy",
                          "resources/minigrid_factored/adv_doorkey_8x8_openyellowdoor_dooryellow_1_termination_negative.npy"]

positive_test_files = ["resources/minigrid_factored/adv_doorkey_8x8_openreddoor_doorred_3_termination_positive.npy"]
negative_test_files = ["resources/minigrid_factored/adv_doorkey_8x8_openreddoor_doorred_3_termination_negative.npy"]


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
        x = x/torch.tensor([7,7,1,5,7,7,7,7,7,7,7,7,7,7,7,7,7,7,4,7,7,7])

        return x
    
    experiment = FactoredAdvancedMinigridDivDisExperiment(base_dir=args.base_dir,
                                                          seed=args.seed,
                                                          policy_phi=policy_phi)
    
    experiment.add_datafiles(positive_train_files,
                             negative_train_files,
                             unlabelled_train_files)
    
    experiment.train_termination(10)
    
    accuracy = experiment.test_terminations(positive_test_files,
                                            negative_test_files)
    
    env = AdvancedDoorKeyPolicyTrainWrapper(
        factored_environment_builder(
            'AdvancedDoorKey-8x8-v0',
            seed=3
        ),
        check_option_complete=check_got_redkey,
        time_limit=50,
        image_input=False
    )
    
    total_steps = 0
    option_rewards = deque(maxlen=100)
    
    while total_steps < 10e6:
        steps, rewards = experiment.perfect_term_policy_train(env=env,
                                                                     idx=0,
                                                                     seed=3)
        
        total_steps += steps
        option_rewards.append(sum(rewards))
        
        print("steps: {} average reward: {} total_reward: {}".format(total_steps,
                                                                   np.mean(option_rewards),
                                                                   np.sum(option_rewards)))
                
    
    # for idx, acc in enumerate(accuracy):
    #     print(idx, acc)
    #     if acc > 0.6:
    #         total_steps = 0
    #         option_rewards = []
    #         env = AdvancedDoorKeyPolicyTrainWrapper(
    #             factored_environment_builder(
    #                 'AdvancedDoorKey-8x8-v0',
    #                 seed=3
    #             ),
    #             door_colour="red",
    #             time_limit=200,
    #             image_input=False
    #         )
    #         while total_steps < 1e6:
    #             steps, option_rewards = experiment.run_rollout(env=env,
    #                                                            idx=idx,
    #                                                            seed=3,
    #                                                            eval=False)
    #             total_steps += steps
    #             option_rewards.append(sum(option_rewards))
                
    #             print("idx {} steps: {} average reward: {}".format(idx,
    #                                                                total_steps,
    #                                                                np.mean(option_rewards)))
                
    #         steps, option_rewards = experiment.run_rollout(env=env,
    #                                                        idx=idx,
    #                                                        seed=3,
    #                                                        eval=True)
            
    #         print("[eval] steps: {} option rewards: {}".format(steps, option_rewards))










