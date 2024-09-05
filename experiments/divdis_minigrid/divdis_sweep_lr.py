import argparse 
from datetime import datetime

from portable.utils.utils import load_gin_configs
from experiments.divdis_minigrid.core.advanced_minigrid_divdis_sweep_experiment import AdvancedMinigridDivDisSweepExperiment

color = 'grey'
task = f'get{color}key'
#task = f'open{color}door'
init_term = 'termination'
RANDOM_TRAIN = True
RANDOM_UNLABELLED = True

base_img_dir = 'resources/minigrid_images/'
positive_train_files = [f"{base_img_dir}adv_doorkey_8x8_v2_{task}_door{color}_0_{init_term}_positive.npy"]
negative_train_files = [f"{base_img_dir}adv_doorkey_8x8_v2_{task}_door{color}_0_{init_term}_negative.npy"]
unlabelled_train_files = [f"{base_img_dir}adv_doorkey_8x8_v2_{task}_door{color}_{s}_{init_term}_{pos_neg}.npy" for s in [1,2] for pos_neg in ['positive', 'negative']]
positive_test_files = [f"{base_img_dir}adv_doorkey_8x8_v2_{task}_door{color}_{s}_{init_term}_positive.npy" for s in [5,6,7,8,9,10]]
negative_test_files = [f"{base_img_dir}adv_doorkey_8x8_v2_{task}_door{color}_{s}_{init_term}_negative.npy" for s in [5,6,7,8,9,10]]

if RANDOM_TRAIN:
    positive_train_files += [f"{base_img_dir}adv_doorkey_8x8_v2_{task}_door{color}_0_1_{init_term}_positive.npy"]
    negative_train_files += [f"{base_img_dir}adv_doorkey_8x8_v2_{task}_door{color}_0_1_{init_term}_negative.npy"]
if RANDOM_UNLABELLED:
    unlabelled_train_files += [f"{base_img_dir}adv_doorkey_8x8_v2_{task}_door{color}_{s}_1_{init_term}_{pos_neg}.npy" for s in [1,2] for pos_neg in ['positive', 'negative']]
positive_test_files += [f"{base_img_dir}adv_doorkey_8x8_v2_{task}_door{color}_{s}_1_{init_term}_positive.npy" for s in [5,6,7,8,9,10,11]]
negative_test_files += [f"{base_img_dir}adv_doorkey_8x8_v2_{task}_door{color}_{s}_1_{init_term}_negative.npy" for s in [5,6,7,8,9,10,11]]

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
    experiment.sweep_lr(-6, # 0.00001
                        -2,
                        10,
                        NUM_SEEDS)


