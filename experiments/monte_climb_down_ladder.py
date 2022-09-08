from portable.environment import MonteBootstrapWrapper
import os
from experiments import Experiment
import numpy as np
from pfrl.wrappers import atari_wrappers

from portable.environment import MonteAgentWrapper
from portable.utils import load_init_states
import argparse

from portable.utils.utils import load_gin_configs

initiation_positive_files = [
    'resources/monte_images/climb_down_ladder_initiation_positive.npy'
]
initiation_negative_files = [
    'resources/monte_images/climb_down_ladder_initiation_negative.npy'
]
termination_positive_files = [
    'resources/monte_images/climb_down_ladder_termination_positive.npy',
    'resources/monte_images/climb_down_ladder_1_termination_positive.npy',
    'resources/monte_images/climb_down_ladder_2_termination_positive.npy',
    'resources/monte_images/climb_down_ladder_3_termination_positive.npy',
    'resources/monte_images/climb_down_ladder_4_termination_positive.npy'
]
termination_negative_files = [
    'resources/monte_images/climb_down_ladder_termination_negative.npy'
]

def phi(x):
    return np.asarray(x, dtype=np.float32) / 255

def make_env(seed):
    env = atari_wrappers.wrap_deepmind(
        atari_wrappers.make_atari('MontezumaRevengeNoFrameskip-v4', max_frames=30*60*60),
        episode_life=True,
        clip_rewards=True,
        frame_stack=False
    )
    env.seed(seed)

    return MonteAgentWrapper(env)

initiation_state_files = [
    [
        'resources/monte_env_states/room1_leftladder_top_0.pkl',
        'resources/monte_env_states/room1_leftladder_top_1.pkl',
        'resources/monte_env_states/room1_leftladder_top_2.pkl',
        'resources/monte_env_states/room1_middleladder_top_0.pkl',
        'resources/monte_env_states/room1_middleladder_top_1.pkl',
        'resources/monte_env_states/room1_middleladder_top_2.pkl',
        'resources/monte_env_states/room1_middleladder_top_3.pkl',
        'resources/monte_env_states/room1_rightladder_top_0.pkl',
        'resources/monte_env_states/room1_rightladder_top_1.pkl',
        'resources/monte_env_states/room1_rightladder_top_2.pkl',
        'resources/monte_env_states/room1_rightladder_top_3.pkl'
    ],
    [
        'resources/monte_env_states/room2_ladder_top_0.pkl',
        'resources/monte_env_states/room2_ladder_top_1.pkl',
        'resources/monte_env_states/room2_ladder_top_2.pkl',
        'resources/monte_env_states/room2_ladder_top_3.pkl',
        'resources/monte_env_states/room2_ladder_top_4.pkl',
        'resources/monte_env_states/room2_ladder_top_5.pkl'
    ],
    [
        'resources/monte_env_states/room7_ladder_top_0.pkl',
        'resources/monte_env_states/room7_ladder_top_1.pkl',
        'resources/monte_env_states/room7_ladder_top_2.pkl',
        'resources/monte_env_states/room7_ladder_top_3.pkl'
    ],
    [
        'resources/monte_env_states/room11_ladder_bottom_0.pkl',
        'resources/monte_env_states/room11_ladder_bottom_1.pkl',
        'resources/monte_env_states/room11_ladder_bottom_2.pkl',
        'resources/monte_env_states/room11_ladder_bottom_3.pkl',
        'resources/monte_env_states/room11_ladder_bottom_4.pkl'
    ],
    [
        'resources/monte_env_states/room13_ladder_bottom_0.pkl',
        'resources/monte_env_states/room13_ladder_bottom_1.pkl',
        'resources/monte_env_states/room13_ladder_bottom_2.pkl',
        'resources/monte_env_states/room13_ladder_bottom_3.pkl'
    ]
]

bootstrap_env = atari_wrappers.wrap_deepmind(
        atari_wrappers.make_atari('MontezumaRevengeNoFrameskip-v4', max_frames=10*60*60),
        episode_life=True,
        clip_rewards=True,
        frame_stack=False
    )
bootstrap_env = MonteBootstrapWrapper(
    bootstrap_env,
    load_init_states(initiation_state_files[0]),
    [
        (21, 148, 1),
        (76, 192, 1),
        (133,148, 1)
    ]
)

trial_names = []

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

    experiment = Experiment(
        base_dir=args.base_dir,
        seed=args.seed,
        policy_phi=phi,
        experiment_env_function=make_env,
        policy_bootstrap_env=bootstrap_env,
        initiation_positive_files=initiation_positive_files,
        initiation_negative_files=initiation_negative_files,
        termination_positive_files=termination_positive_files,
        termination_negative_files=termination_negative_files
    )

    experiment.save()

    # experiment.run_trial(

    # )
    
