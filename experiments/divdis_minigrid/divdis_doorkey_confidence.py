from experiments.divdis_minigrid.core.advanced_minigrid_factored_divdis_classifier_experiment import AdvancedMinigridFactoredDivDisClassifierExperiment
import argparse
from portable.utils.utils import load_gin_configs

positive_train_files = ["resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_0_initiation_positive.npy"]
negative_train_files = ["resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_0_initiation_negative.npy"]
unlabelled_train_files = ["resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_1_initiation_positive.npy",
                          "resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_1_initiation_negative.npy",
                          "resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_2_initiation_positive.npy",
                          "resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_2_initiation_negative.npy"]

positive_test_files = [
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_3_initiation_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_4_initiation_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_5_initiation_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_6_initiation_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_7_initiation_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_8_initiation_positive.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_9_initiation_positive.npy",
                      ]
negative_test_files = [
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_3_initiation_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_4_initiation_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_5_initiation_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_6_initiation_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_7_initiation_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_8_initiation_negative.npy",
    "resources/factored_minigrid_images/adv_doorkey_8x8_v2_openreddoor_doorred_9_initiation_negative.npy",
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

        experiment = AdvancedMinigridFactoredDivDisClassifierExperiment(base_dir=args.base_dir,
                                                                        seed=args.seed)

        experiment.add_datafiles(positive_train_files,
                                 negative_train_files,
                                 unlabelled_train_files)

        experiment.train_classifier(200)
        
        experiment.test_confidences(positive_test_files,
                                    negative_test_files)
