from experiments.core.divdis_meta_experiment import DivDisMetaExperiment
import argparse
from portable.utils.utils import load_gin_configs, load_init_states
import torch 
from portable.agent.model.ppo import create_atari_model
from experiments.monte.environment import MonteBootstrapWrapper, MonteAgentWrapper
from pfrl.wrappers import atari_wrappers
from experiments.divdis_monte.core.monte_terminations import *
from experiments.divdis_monte.experiment_files import *
import numpy as np

terminations = [
    [check_termination_bottom_ladder],
    [check_termination_top_ladder],
    [check_termination_correct_enemy_left],
    [check_termination_correct_enemy_right]
]

def make_bootstrap_env(init_states, termination_func, term_points):
    env = atari_wrappers.wrap_deepmind(
        atari_wrappers.make_atari('MontezumaRevengeNoFrameskip-v4'),
        episode_life=True,
        clip_rewards=True,
        frame_stack=False
    )
    env.seed(0)
    env = MonteBootstrapWrapper(env,
                                agent_space=False,
                                list_init_states=load_init_states(init_states),
                                check_true_termination=termination_func,
                                list_termination_points=term_points,
                                max_steps=500)
    return env

bootstrap_envs = [
    [[make_bootstrap_env(
        ['resources/monte_env_states/room1/ladder/left_top_0.pkl',
         'resources/monte_env_states/room1/ladder/left_top_1.pkl',
         'resources/monte_env_states/room1/ladder/left_top_2.pkl',
         'resources/monte_env_states/room1/ladder/middle_top_0.pkl',
         'resources/monte_env_states/room1/ladder/middle_top_1.pkl',
         'resources/monte_env_states/room1/ladder/middle_top_2.pkl',
         'resources/monte_env_states/room1/ladder/middle_top_3.pkl',
         'resources/monte_env_states/room1/ladder/right_top_0.pkl',
         'resources/monte_env_states/room1/ladder/right_top_1.pkl',
         'resources/monte_env_states/room1/ladder/right_top_2.pkl',
         'resources/monte_env_states/room1/ladder/right_top_3.pkl'],
        epsilon_ball_termination,
        [
            (20, 148, 1),
            (20, 148, 1),
            (20, 148, 1),
            (77, 192, 1),
            (77, 192, 1),
            (77, 192, 1),
            (77, 192, 1),
            (133,148, 1),
            (133,148, 1),
            (133,148, 1),
            (133,148, 1)
        ])]],
    [[make_bootstrap_env(
        ['resources/monte_env_states/room1/ladder/left_bottom_0.pkl',
         'resources/monte_env_states/room1/ladder/left_bottom_1.pkl',
         'resources/monte_env_states/room1/ladder/left_bottom_2.pkl',
         'resources/monte_env_states/room1/ladder/middle_bottom_0.pkl',
         'resources/monte_env_states/room1/ladder/middle_bottom_1.pkl',
         'resources/monte_env_states/room1/ladder/middle_bottom_2.pkl',
         'resources/monte_env_states/room1/ladder/right_bottom_0.pkl',
         'resources/monte_env_states/room1/ladder/right_bottom_1.pkl',
         'resources/monte_env_states/room1/ladder/right_bottom_2.pkl',],
        epsilon_ball_termination,
        [
            (20, 192, 1),
            (20, 192, 1),
            (20, 192, 1),
            (77, 235, 1),
            (77, 235, 1),
            (77, 235, 1),
            (133, 192, 1),
            (133, 192, 1),
            (133, 192, 1),
        ])]],
    [[make_bootstrap_env(
        ['resources/monte_env_states/room1/enemy/skull_right_0.pkl',
         'resources/monte_env_states/room1/enemy/skull_right_1.pkl',
         'resources/monte_env_states/room1/ladder/right_bottom_0.pkl',
         'resources/monte_env_states/room1/ladder/right_bottom_1.pkl',
         'resources/monte_env_states/room1/ladder/right_bottom_2.pkl',],
        enemy_left_termination,
        [(),
         (),
         (),
         (),
         ()])]],
    [[make_bootstrap_env(
        ['resources/monte_env_states/room1/enemy/skull_left_0.pkl',
         'resources/monte_env_states/room1/enemy/skull_left_1.pkl',
         'resources/monte_env_states/room1/ladder/left_bottom_0.pkl',
         'resources/monte_env_states/room1/ladder/left_bottom_1.pkl',
         'resources/monte_env_states/room1/ladder/left_bottom_2.pkl',],
        enemy_right_termination,
        [(),
         (),
         (),
         (),
         ()])]],
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
    
    experiment = DivDisMetaExperiment(base_dir=args.base_dir,
                                      seed=args.seed,
                                      option_policy_phi=policy_phi,
                                      agent_phi=option_agent_phi,
                                      action_model=create_atari_model(4,22),
                                      option_type="mock",
                                      terminations=terminations)
    
    experiment.train_option_policies(
        bootstrap_envs,
        0,
        max_steps=1e6
    )
    
    meta_env = atari_wrappers.wrap_deepmind(
        atari_wrappers.make_atari('MontezumaRevengeNoFrameskip-v4', max_frames=1000),
        episode_life=True,
        clip_rewards=True,
        frame_stack=False
    )
    meta_env.seed(args.seed)
    
    meta_env = MonteAgentWrapper(meta_env, agent_space=False)
    
    experiment.train_meta_agent(meta_env,
                                args.seed,
                                4e6)



