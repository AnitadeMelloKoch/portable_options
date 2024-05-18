from experiments.core.divdis_meta_experiment import DivDisMetaExperiment
import argparse
from portable.utils.utils import load_gin_configs
from portable.agent.model.ppo import create_atari_model
import numpy as np
import torch

from experiments.monte.environment import MonteBootstrapWrapper, MonteAgentWrapper
from portable.utils import load_init_states
from pfrl.wrappers import atari_wrappers
from experiments.divdis_monte.core.monte_terminations import *
from experiments.divdis_monte.experiment_files import *

init_states = [
    ["resources/monte_env_states/room0/ladder/top_0.pkl"],
    ["resources/monte_env_states/room11/enemy/left_of_left_snake.pkl"],
    ["resources/monte_env_states/room21/platforms/left.pkl"]
]

termination_point = [
    [(5, 235, 4)],
    [(77, 235, 19)],
    [(79, 235, 7)]
]

# climb down ladder
# climb up ladder
# move left enemy
# move right enemy

positive_files = [
    ["resources/monte_images/screen_climb_down_ladder_termination_positive.npy"],
    ["resources/monte_images/climb_up_ladder_room1_termination_positive.npy"],
    ["resources/monte_images/move_left_enemy_room1_termination_positive.npy"],
    ["resources/monte_images/move_right_enemy_room1_termination_positive.npy"],
]
negative_files = [
    ["resources/monte_images/screen_climb_down_ladder_termination_negative.npy",
     "resources/monte_images/screen_death_1.npy",
     "resources/monte_images/screen_death_2.npy",
     "resources/monte_images/screen_death_3.npy",
     "resources/monte_images/screen_death_4.npy"],
    ["resources/monte_images/climb_up_ladder_room1_termination_negative.npy",
     "resources/monte_images/screen_death_1.npy",
     "resources/monte_images/screen_death_2.npy",
     "resources/monte_images/screen_death_3.npy",
     "resources/monte_images/screen_death_4.npy"],
    ["resources/monte_images/move_left_enemy_room1_termination_negative.npy",
     "resources/monte_images/screen_death_1.npy",
     "resources/monte_images/screen_death_2.npy",
     "resources/monte_images/screen_death_3.npy",
     "resources/monte_images/screen_death_4.npy"],
    ["resources/monte_images/move_right_enemy_room1_termination_negative.npy",
     "resources/monte_images/screen_death_1.npy",
     "resources/monte_images/screen_death_2.npy",
     "resources/monte_images/screen_death_3.npy",
     "resources/monte_images/screen_death_4.npy"],
]
unlabelled_files = [
    ["resources/monte_images/climb_down_ladder_room0_initiation_positive.npy",
     "resources/monte_images/climb_down_ladder_room0_initiation_positive.npy",
     "resources/monte_images/move_left_enemy_room11left_termination_negative.npy",
     "resources/monte_images/move_left_enemy_room11left_termination_positive.npy",],
    ["resources/monte_images/climb_down_ladder_room0_initiation_positive.npy",
     "resources/monte_images/climb_down_ladder_room0_initiation_positive.npy",
     "resources/monte_images/move_left_enemy_room11left_termination_negative.npy",
     "resources/monte_images/move_left_enemy_room11left_termination_positive.npy",],
    ["resources/monte_images/climb_down_ladder_room0_initiation_positive.npy",
     "resources/monte_images/climb_down_ladder_room0_initiation_positive.npy",
     "resources/monte_images/move_left_enemy_room11left_termination_negative.npy",
     "resources/monte_images/move_left_enemy_room11left_termination_positive.npy",],
    ["resources/monte_images/climb_down_ladder_room0_initiation_positive.npy",
     "resources/monte_images/climb_down_ladder_room0_initiation_positive.npy",
     "resources/monte_images/move_left_enemy_room11left_termination_negative.npy",
     "resources/monte_images/move_left_enemy_room11left_termination_positive.npy",],
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
    # monte has 18 actions
    
    for idx in range(len(init_states)):
    
        experiment = DivDisMetaExperiment(base_dir=args.base_dir,
                                        seed=args.seed,
                                        option_policy_phi=policy_phi,
                                        agent_phi=option_agent_phi,
                                        action_model=create_atari_model(4, 13),
                                        option_type="divdis",
                                        add_unlabelled_data=True)
        
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
                                    list_init_states=load_init_states(init_states[idx]),
                                    check_true_termination=epsilon_ball_termination,
                                    list_termination_points=termination_point[idx],
                                    max_steps=int(2e4))
        
        experiment.add_datafiles(positive_files=positive_files,
                                negative_files=negative_files,
                                unlabelled_files=unlabelled_files)
        
        experiment.train_option_classifiers()
        
        experiment.train_meta_agent(env,
                                    args.seed,
                                    2e6,
                                    0.9)