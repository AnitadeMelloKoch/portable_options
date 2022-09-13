from portable.environment import MonteBootstrapWrapper
import os
from experiments import Experiment
import numpy as np
from pfrl.wrappers import atari_wrappers

from portable.environment import MonteAgentWrapper
from portable.utils import load_init_states
import argparse

from portable.utils.utils import load_gin_configs

import torch

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
        atari_wrappers.make_atari('MontezumaRevengeNoFrameskip-v4', max_frames=2*60*60),
        episode_life=True,
        clip_rewards=True,
        frame_stack=False
    )
    env.seed(seed)

    return MonteAgentWrapper(env, agent_space=True)

initiation_state_files = [
    [
        'resources/monte_env_states/room1/ladder/left_top_0.pkl',
        'resources/monte_env_states/room1/ladder/left_top_1.pkl',
        'resources/monte_env_states/room1/ladder/left_top_2.pkl',
        'resources/monte_env_states/room1/ladder/middle_top_0.pkl',
        'resources/monte_env_states/room1/ladder/middle_top_1.pkl',
        'resources/monte_env_states/room1/ladder/middle_top_2.pkl',
        'resources/monte_env_states/room1/ladder/middle_top_3.pkl',
        'resources/monte_env_states/room1/ladder/right_top_0.pkl',
        'resources/monte_env_states/room1/ladder/right_top_1.pkl',
        'resources/monte_env_states/room1/ladder/right_top_2.pkl',
        'resources/monte_env_states/room1/ladder/right_top_3.pkl'
    ],[
        'resources/monte_env_states/room0/ladder/top_0.pkl',
        'resources/monte_env_states/room0/ladder/top_1.pkl',
        'resources/monte_env_states/room0/ladder/top_2.pkl',
        'resources/monte_env_states/room0/ladder/top_3.pkl'
    ],[
        'resources/monte_env_states/room2/ladder/top_0.pkl',
        'resources/monte_env_states/room2/ladder/top_1.pkl',
        'resources/monte_env_states/room2/ladder/top_2.pkl'
    ],[
        'resources/monte_env_states/room7/ladder/top_0.pkl',
        'resources/monte_env_states/room7/ladder/top_1.pkl',
        'resources/monte_env_states/room7/ladder/top_2.pkl'
    ],[
        'resources/monte_env_states/room11/ladder/bottom_0.pkl',
        'resources/monte_env_states/room11/ladder/bottom_1.pkl',
        'resources/monte_env_states/room11/ladder/bottom_2.pkl',
        'resources/monte_env_states/room11/ladder/bottom_3.pkl',
    ],[
        'resources/monte_env_states/room13/ladder/bottom_0.pkl',
        'resources/monte_env_states/room13/ladder/bottom_1.pkl',
        'resources/monte_env_states/room13/ladder/bottom_2.pkl',
        'resources/monte_env_states/room13/ladder/bottom_3.pkl'
        'resources/monte_env_states/room13/ladder/bottom_4.pkl'
        'resources/monte_env_states/room13/ladder/bottom_5.pkl'
    ],[
        'resources/monte_env_states/room4/ladder/bottom_0.pkl',
        'resources/monte_env_states/room4/ladder/bottom_1.pkl',
        'resources/monte_env_states/room4/ladder/bottom_2.pkl'
    ],[
        'resources/monte_env_states/room3/ladder/top_0.pkl',
        'resources/monte_env_states/room3/ladder/top_1.pkl',
        'resources/monte_env_states/room3/ladder/top_2.pkl',
    ],[
        'resources/monte_env_states/room5/ladder/top_0.pkl',
        'resources/monte_env_states/room5/ladder/top_1.pkl',
        'resources/monte_env_states/room5/ladder/top_2.pkl',
        'resources/monte_env_states/room5/ladder/top_3.pkl',
    ],[
        'resources/monte_env_states/room14/ladder/top_0.pkl',
        'resources/monte_env_states/room14/ladder/top_1.pkl',
        'resources/monte_env_states/room14/ladder/top_2.pkl',
    ]
]

terminations = [
    [
        [(21, 148, 1),(20, 148, 1),(22, 148, 1)],
        [(21, 148, 1),(20, 148, 1),(22, 148, 1)],
        [(21, 148, 1),(20, 148, 1),(22, 148, 1)],
        [(76, 192, 1),(77, 192, 1),(78, 192, 1)],
        [(76, 192, 1),(77, 192, 1),(78, 192, 1)],
        [(76, 192, 1),(77, 192, 1),(78, 192, 1)],
        [(76, 192, 1),(77, 192, 1),(78, 192, 1)],
        [(133,148, 1),(134,148, 1),(135,148, 1)],
        [(133,148, 1),(134,148, 1),(135,148, 1)],
        [(133,148, 1),(134,148, 1),(135,148, 1)],
        [(133,148, 1),(134,148, 1),(135,148, 1)]
    ],[
        [(77, 237,4), (76, 235,4), (80, 235, 4)],
        [(77, 237,4), (76, 235,4), (80, 235, 4)],
        [(77, 237,4), (76, 235,4), (80, 235, 4)]
    ],[
        [(77,235,6)],
        [(77,235,6)],
        [(77,235,6)]
    ],[
        [(77,237,13), (80,235,13),(74,235,13),(77,233,13)],
        [(77,237,13), (80,235,13),(74,235,13),(77,233,13)],
        [(77,237,13), (80,235,13),(74,235,13),(77,233,13)],
        [(77,237,13), (80,235,13),(74,235,13),(77,233,13)],
    ],[
        [(77,235,19)],
        [(77,235,19)],
        [(77,235,19)],
        [(77,235,19)],
    ],[
        [(76,235,21),(80,235,21),(72,235,21),(77,235,21)],
        [(76,235,21),(80,235,21),(72,235,21),(77,235,21)],
        [(76,235,21),(80,235,21),(72,235,21),(77,235,21)],
        [(76,235,21),(80,235,21),(72,235,21),(77,235,21)],
    ],[
        [(77,235,10)],
        [(77,235,10)],
        [(77,235,10)]
    ],[
        [(77,235,9)],
        [(77,235,9)],
        [(77,235,9)],
        [(77,235,9)],
    ],[
        [(77,235,11)],
        [(77,235,11)],
        [(77,235,11)],
        [(77,235,11)],
    ],[
        [(77,235,22)],
        [(77,235,22)],
        [(77,235,22)],
        [(77,235,22)],
    ]
]

room_names = [
    'room1',
    'room0',
    'room2',
    'room7',
    'room11',
    'room13',
    'room4',
    'room3',
    'room5',
    'room14',
]

bootstrap_env = atari_wrappers.wrap_deepmind(
        atari_wrappers.make_atari('MontezumaRevengeNoFrameskip-v4', max_frames=2*60*60),
        episode_life=True,
        clip_rewards=True,
        frame_stack=False
    )
bootstrap_env = MonteBootstrapWrapper(
    bootstrap_env,
    load_init_states(initiation_state_files[0]),
    terminations[0],
    agent_space=True
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

    # experiment.load()
    experiment.save()

    # experiment.bootstrap_from_room(
    #     load_init_states(initiation_state_files[0]),
    #     terminations[0],
    #     1000
    # )

    # # experiment.save()

    # experiment.run_trial(
    #     load_init_states(initiation_state_files[0]),
    #     terminations[0],
    #     1000,
    #     eval=True,
    #     trial_name="room1_eval"
    # )

    # for x in range(len(initiation_state_files)):
    #     experiment.run_trial(
    #         load_init_states(initiation_state_files[x]),
    #         terminations[x],
    #         100000,
    #         eval=False,
    #         trial_name="{}_train".format(room_names[x])
    #     )
    #     for y in range(len(initiation_state_files)):
    #         experiment.run_trial(
    #             load_init_states(initiation_state_files[x]),
    #             terminations[x],
    #             10000,
    #             eval=True,
    #             trial_name="{}_eval_after_{}_train".format(room_names[y], room_names[x])
    #         )
    
