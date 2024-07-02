from experiments.core.divdis_option_experiment import DivDisOptionExperiment
import argparse 
from portable.utils.utils import load_gin_configs
import numpy as np
import torch
import os

from experiments.monte.environment import MonteBootstrapWrapper, MonteAgentWrapper
from portable.utils import load_init_states
from pfrl.wrappers import atari_wrappers
from experiments.divdis_monte.core.monte_terminations import *
from experiments.divdis_monte.experiment_files import *

init_states = [
    ["resources/monte_env_states/room0/ladder/top_0.pkl",
     "resources/monte_env_states/room0/ladder/top_1.pkl",
     "resources/monte_env_states/room0/ladder/top_2.pkl",
     "resources/monte_env_states/room0/ladder/top_3.pkl",],
    ["resources/monte_env_states/room1/ladder/middle_top_0.pkl",
     "resources/monte_env_states/room1/ladder/left_top_0.pkl",
     "resources/monte_env_states/room1/ladder/right_top_0.pkl",],
    ["resources/monte_env_states/room2/ladder/top_0.pkl",
     "resources/monte_env_states/room2/ladder/top_1.pkl",
     "resources/monte_env_states/room2/ladder/top_2.pkl",],
    ["resources/monte_env_states/room3/ladder/top_0.pkl"],
    ["resources/monte_env_states/room5/ladder/top_0.pkl",
     "resources/monte_env_states/room5/ladder/top_1.pkl",
     "resources/monte_env_states/room5/ladder/top_2.pkl",],
    ["resources/monte_env_states/room7/ladder/top_0.pkl",
     "resources/monte_env_states/room7/ladder/top_1.pkl",
     "resources/monte_env_states/room7/ladder/top_2.pkl",],
    ["resources/monte_env_states/room14/ladder/top_0.pkl",
     "resources/monte_env_states/room14/ladder/top_1.pkl",
     "resources/monte_env_states/room14/ladder/top_2.pkl"]
]

# files for each room
positive_files = [
    ["resources/monte_images/climb_down_ladder_room10_termination_positive.npy"],
    ["resources/monte_images/screen_climb_down_ladder_termination_positive.npy"],
    ["resources/monte_images/climb_down_ladder_room6_termination_positive.npy"],
    ["resources/monte_images/climb_down_ladder_room9_termination_positive.npy"],
    ["resources/monte_images/climb_down_ladder_room21_termination_positive.npy"],
    ["resources/monte_images/climb_down_ladder_room19_termination_positive.npy"],
    ["resources/monte_images/climb_down_ladder_room22_termination_positive.npy"]
]
negative_files = [
    ["resources/monte_images/climb_down_ladder_room0_initiation_negative.npy",
     "resources/monte_images/climb_down_ladder_room4_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room10_termination_negative.npy"],
    ["resources/monte_images/screen_climb_down_ladder_termination_negative.npy"],
    ["resources/monte_images/climb_down_ladder_room2_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room6_termination_negative.npy"],
    ["resources/monte_images/climb_down_ladder_room3_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room9_termination_negative.npy"],
    ["resources/monte_images/climb_down_ladder_room7_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room13_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room21_termination_negative.npy"],
    ["resources/monte_images/climb_down_ladder_room5_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room11_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room19_termination_negative.npy"],
    ["resources/monte_images/climb_down_ladder_room14_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room22_termination_negative.npy"],
]
unlabelled_files = [
    ["resources/monte_images/climb_down_ladder_room9_termination_positive.npy",
     "resources/monte_images/climb_down_ladder_room9_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room10_uncertain.npy",
     "resources/monte_images/lasers_wait_disappear_room7_termination_positive.npy",
     "resources/monte_images/lasers_wait_disappear_room7_termination_negative.npy",],
    ["resources/monte_images/lasers_wait_to_disappear_room12_termination_positive.npy",
     "resources/monte_images/lasers_wait_to_disappear_room12_termination_positive.npy",
     "resources/monte_images/climb_down_ladder_room7_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room13_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room21_termination_negative.npy"],
    ["resources/monte_images/climb_down_ladder_room21_termination_positive.npy",
     "resources/monte_images/climb_down_ladder_room13_uncertain.npy",
     "resources/monte_images/climb_down_ladder_room7_uncertain.npy"],
    ["resources/monte_images/climb_down_ladder_room3_uncertain.npy",
     "resources/monte_images/climb_down_ladder_room9_uncertain.npy"],
    ["resources/monte_images/climb_down_ladder_room2_uncertain.npy"],
    ["resources/monte_images/climb_down_ladder_room21_uncertain.npy"],
    ["resources/monte_images/climb_down_ladder_room19_uncertain.npy"]
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--sub_dir", type=str, default="")
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
    
    if args.sub_dir == "":
        base_dir = os.path.join(args.base_dir, args.sub_dir)
    else:
        base_dir = args.base_dir
    
    experiment = DivDisOptionExperiment(base_dir=base_dir,
                                        seed=args.seed,
                                        option_type="divdis",
                                        policy_phi=policy_phi)
    
    file_idx = 0
    
    for pos, neg, unlab in zip(positive_files,negative_files,unlabelled_files):
        experiment.option.reset_classifiers()
        experiment.add_datafiles(pos, neg, unlab)
        experiment.train_classifier()
        for state_idx, init_state in enumerate(init_states):
            experiment.change_option_save(name="option_files{}_state{}".format(file_idx,
                                                                               state_idx))
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
                                        list_termination_points=[(0,0,0)]*len(init_state),
                                        max_steps=int(2e4))
            experiment.train_option(env,
                                    args.seed,
                                    5e5,
                                    state_idx)
    
    experiment.save()

