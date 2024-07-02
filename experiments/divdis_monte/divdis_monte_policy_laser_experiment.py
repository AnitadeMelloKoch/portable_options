from experiments.core.divdis_option_experiment import DivDisOptionExperiment
import argparse 
from portable.utils.utils import load_gin_configs
import numpy as np
import torch

from experiments.monte.environment import MonteBootstrapWrapper, MonteAgentWrapper
from portable.utils import load_init_states
from pfrl.wrappers import atari_wrappers
from experiments.divdis_monte.core.monte_terminations import *
from experiments.divdis_monte.experiment_files import *

init_states = [
    ["resources/monte_env_states/room0/ladder/top_0.pkl"],
    ["resources/monte_env_states/room0/lasers/left_with_laser.pkl"],
    ["resources/monte_env_states/room7/ladder/top_0.pkl"],
    ["resources/monte_env_states/room7/lasers/between_right_lasers.pkl"]
]

# files for each room
positive_files = [
    ["resources/monte_images/lasers1_toleft_room0_termination_positive.npy"],
    ["resources/monte_images/lasers2_toleft_room0_termination_positive.npy"]
]
negative_files = [
    ["resources/monte_images/lasers1_toleft_room0_termination_negative.npy"],
    ["resources/monte_images/lasers2_toleft_room0_termination_negative.npy"],
]
unlabelled_files = [
    ["resources/monte_images/lasers2_toleft_room0_termination_positive.npy",
     "resources/monte_images/lasers2_toleft_room0_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room2_initiation_positive.npy",
     "resources/monte_images/lasers_wait_disappear_room7_termination_positive.npy",
     "resources/monte_images/lasers_wait_disappear_room7_termination_negative.npy",],
    ["resources/monte_images/climb_down_ladder_room2_initiation_positive.npy",
     "resources/monte_images/lasers_wait_disappear_room7_termination_positive.npy",
     "resources/monte_images/lasers_wait_disappear_room7_termination_negative.npy",]
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
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        x = (x/255.0).float()
        return x
    
    def option_agent_phi(x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        x = (x/255.0).float()
        return x
    
    experiment = DivDisOptionExperiment(base_dir=args.base_dir,
                                        seed=args.seed,
                                        option_type="divdis",
                                        policy_phi=policy_phi)
    
    file_idx = 0
    
    for pos, neg, unlab in zip(positive_files,negative_files,unlabelled_files):
        experiment.option.reset_classifiers()
        experiment.add_datafiles(pos, neg, unlab)
        experiment.train_classifier()
        for env_idx, init_state in enumerate(init_states):
            experiment.change_option_save(name="option_files{}_state{}".format(file_idx,
                                                                               env_idx))
            file_idx += 1
            env = atari_wrappers.wrap_deepmind(
                atari_wrappers.make_atari('MontezumaRevengeNoFrameskip-v4', max_frames=1000),
                episode_life=True,
                clip_rewards=True,
                frame_stack=False
            )
            env.seed(args.seed)

            env = MonteAgentWrapper(env, agent_space=False, stack_observations=False)
            env = MonteBootstrapWrapper(env,
                                        agent_space=False,
                                        list_init_states=load_init_states(init_state),
                                        check_true_termination=lambda x,y,z: False,
                                        list_termination_points=[(0,0,0)],
                                        max_steps=int(2e4))
            experiment.train_option(env,
                                    args.seed,
                                    5e5,
                                    env_idx)
    
    experiment.save()

