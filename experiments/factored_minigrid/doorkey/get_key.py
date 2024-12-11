from experiments.factored_minigrid.experiment import FactoredMinigridSingleOptionExperiment
from experiments.factored_minigrid.utils import environment_builder
import argparse
from experiments.factored_minigrid.doorkey.core.policy_train_wrapper import FactoredDoorKeyPolicyTrainWrapper, get_key
from portable.utils import load_gin_configs
import numpy as np 

initiation_positive_files = [
    'resources/minigrid_images/doorkey_getkey_1_initiation_positive.npy'
]

initiation_negative_files = [
    'resources/minigrid_images/doorkey_getkey_1_initiation_negative.npy'
]

termination_positive_files = [
    'resources/minigrid_images/doorkey_getkey_1_termination_positive.npy'
]

termination_negative_files = [
    'resources/minigrid_images/doorkey_getkey_1_termination_negative.npy'
]

train_seed = 0

def policy_phi(x):
    if np.max(x) > 1:
        x = x/255.0
    x = x.astype(np.float32)
    return x

def create_env(seed):
    return environment_builder('FactoredMiniGrid-DoorKey-8x8-v0',
                               seed=seed)

def get_latent_state_function(env, info):
    return

def markov_option_builder(todo_check_args):
    return 

train_env = FactoredDoorKeyPolicyTrainWrapper(
    environment_builder('FactoredMiniGrid-DoorKey-8x8-v0',
                        seed=train_seed),
    get_key
)

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
    
    experiment = FactoredMinigridSingleOptionExperiment(base_dir=args.base_dir,
                                                        random_seed=args.seed,
                                                        create_env_function=create_env,
                                                        policy_phi=policy_phi,
                                                        training_env=[train_env],
                                                        get_latent_state_function=get_latent_state_function,
                                                        markov_option_builder=markov_option_builder,
                                                        initiation_positive_files=initiation_positive_files,
                                                        initiation_negative_files=initiation_negative_files,
                                                        termination_positive_files=termination_positive_files,
                                                        termination_negative_files=termination_negative_files)

