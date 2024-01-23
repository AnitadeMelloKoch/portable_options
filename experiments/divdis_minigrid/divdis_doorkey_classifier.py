from experiments.divdis_minigrid.core.advanced_minigrid_factored_divdis_experiment import AdvancedMinigridFactoredDivDisExperiment
import argparse 
from portable.utils.utils import load_gin_configs
import torch 
import random 

positive_train_files = ["resources/minigrid_factored/adv_doorkey_8x8_openreddoor_doorred_2_initiation_positive.npy"]
negative_train_files = ["resources/minigrid_factored/adv_doorkey_8x8_openreddoor_doorred_2_initiation_negative.npy"]
unlabelled_train_files = ["resources/minigrid_factored/adv_doorkey_8x8_openbluedoor_doorblue_1_initiation_positive.npy",
                          "resources/minigrid_factored/adv_doorkey_8x8_openbluedoor_doorblue_0_initiation_negative.npy",
                          "resources/minigrid_factored/adv_doorkey_8x8_openyellowdoor_dooryellow_2_initiation_positive.npy",
                          "resources/minigrid_factored/adv_doorkey_8x8_openyellowdoor_dooryellow_1_initiation_negative.npy"]

positive_test_files = ["resources/minigrid_factored/adv_doorkey_8x8_openreddoor_doorred_3_initiation_positive.npy"]
negative_test_files = ["resources/minigrid_factored/adv_doorkey_8x8_openreddoor_doorred_3_initiation_negative.npy"]

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

        experiment = AdvancedMinigridFactoredDivDisExperiment(base_dir=args.base_dir,
                                                                seed=args.seed)

        experiment.add_datafiles(positive_train_files,
                                 negative_train_files,
                                 unlabelled_train_files)

        experiment.train_classifier(100)
        
        accuracy = experiment.test_classifier(positive_test_files,
                                              negative_test_files)
        
        print(accuracy)
        
        # true_stats, false_stats = experiment.explain_classifiers(unlabelled_train_files,
        #                                                          7)

        # print(true_stats[1]*torch.tensor([7,7,1,5,7,7,7,7,7,7,7,7,7,7,7,7,7,7,4,7,7,7]))
        # print(true_stats[0])
        # print(false_stats[1]*torch.tensor([7,7,1,5,7,7,7,7,7,7,7,7,7,7,7,7,7,7,4,7,7,7]))
        # print(false_stats[0])
