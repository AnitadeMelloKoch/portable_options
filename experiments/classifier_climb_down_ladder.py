import os
from experiments import ClassifierExperiment
import numpy as np
from pfrl.wrappers import atari_wrappers

from portable.environment import MonteAgentWrapper, MonteBootstrapWrapper
from portable.utils import load_init_states
import argparse

from portable.utils.utils import load_gin_configs

def make_env(seed):
    env = atari_wrappers.wrap_deepmind(
        atari_wrappers.make_atari('MontezumaRevengeNoFrameskip-v4', max_frames=1000),
        episode_life=True,
        clip_rewards=True,
        frame_stack=False
    )
    env.seed(seed)

    return MonteAgentWrapper(env, agent_space=False)

initiation_positive_files = [
    'resources/monte_images/climb_down_ladder_initiation_positive.npy'
]
initiation_negative_files = [
    'resources/monte_images/climb_down_ladder_initiation_negative.npy'
]
initiation_priority_negative_files = [
    'resources/monte_images/death.npy',
    'resources/monte_images/falling_1.npy',
    'resources/monte_images/falling_2.npy',
    'resources/monte_images/falling_3.npy',
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
termination_priority_negative_files = [
    'resources/monte_images/climb_down_ladder_initiation_positive.npy',
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

    experiment = ClassifierExperiment(
        base_dir=args.base_dir,
        seed=args.seed,
        options_initiation_positive_files=initiation_positive_files,
        options_initiation_negative_files=initiation_negative_files,
        options_initiation_priority_negative_files=initiation_priority_negative_files,
        options_termination_positive_files=termination_positive_files,
        options_termination_negative_files=termination_negative_files,
        options_termination_priority_negative_files=termination_priority_negative_files
    )

    # experiment.to_right_room(
    #     make_env(args.seed)
    # )

    bootstrap_env = atari_wrappers.wrap_deepmind(
        atari_wrappers.make_atari('MontezumaRevengeNoFrameskip-v4', max_frames=1000),
        episode_life=True,
        clip_rewards=True,
        frame_stack=False
    )
    # bootstrap_env = MonteBootstrapWrapper(
    #     bootstrap_env,
    #     load_init_states(['resources/monte_env_states/room0/ladder/top_0.pkl']),
    #     [(77, 237,4), (76, 235,4), (80, 235, 4)],
    #     lambda x,y,z: False,
    #     agent_space=False
    # )
    # experiment.room_0(
    #     bootstrap_env
    # )

    # bootstrap_env = MonteBootstrapWrapper(
    #     bootstrap_env,
    #     load_init_states(['resources/monte_env_states/room2/ladder/top_0.pkl']),
    #     [(77, 237,4), (76, 235,4), (80, 235, 4)],
    #     lambda x,y,z: False,
    #     agent_space=False
    # )
    # experiment.room_2(
    #     bootstrap_env
    # )

    # bootstrap_env = MonteBootstrapWrapper(
    #     bootstrap_env,
    #     load_init_states(['resources/monte_env_states/room3/ladder/top_0.pkl']),
    #     [(77, 237,4), (76, 235,4), (80, 235, 4)],
    #     lambda x,y,z: False,
    #     agent_space=False
    # )
    # experiment.room_3(
    #     bootstrap_env
    # )

    # bootstrap_env = MonteBootstrapWrapper(
    #     bootstrap_env,
    #     load_init_states(['resources/monte_env_states/room4/ladder/bottom_0.pkl']),
    #     [(77, 237,4), (76, 235,4), (80, 235, 4)],
    #     lambda x,y,z: False,
    #     agent_space=False
    # )
    # experiment.room_4(
    #     bootstrap_env
    # )

    # bootstrap_env = MonteBootstrapWrapper(
    #     bootstrap_env,
    #     load_init_states(['resources/monte_env_states/room5/ladder/top_3.pkl']),
    #     [(77, 237,4), (76, 235,4), (80, 235, 4)],
    #     lambda x,y,z: False,
    #     agent_space=False
    # )
    # experiment.room_5(
    #     bootstrap_env
    # )

    # bootstrap_env = MonteBootstrapWrapper(
    #     bootstrap_env,
    #     load_init_states(['resources/monte_env_states/room6/ladder/bottom_0.pkl']),
    #     [(77, 237,4), (76, 235,4), (80, 235, 4)],
    #     lambda x,y,z: False,
    #     agent_space=False
    # )
    # experiment.room_6(
    #     bootstrap_env
    # )

    # bootstrap_env = MonteBootstrapWrapper(
    #     bootstrap_env,
    #     load_init_states(['resources/monte_env_states/room7/ladder/top_0.pkl']),
    #     [(77, 237,4), (76, 235,4), (80, 235, 4)],
    #     lambda x,y,z: False,
    #     agent_space=False
    # )
    # experiment.room_7(
    #     bootstrap_env
    # )

    # bootstrap_env = MonteBootstrapWrapper(
    #     bootstrap_env,
    #     load_init_states(['resources/monte_env_states/room9/ladder/bottom_0.pkl']),
    #     [(77, 237,4), (76, 235,4), (80, 235, 4)],
    #     lambda x,y,z: False,
    #     agent_space=False
    # )
    # experiment.room_9(
    #     bootstrap_env
    # )

    # bootstrap_env = MonteBootstrapWrapper(
    #     bootstrap_env,
    #     load_init_states(['resources/monte_env_states/room10/ladder/bottom_0.pkl']),
    #     [(77, 237,4), (76, 235,4), (80, 235, 4)],
    #     lambda x,y,z: False,
    #     agent_space=False
    # )
    # experiment.room_10(
    #     bootstrap_env
    # )

    bootstrap_env = MonteBootstrapWrapper(
        bootstrap_env,
        load_init_states(['resources/monte_env_states/room11/ladder/bottom_0.pkl']),
        [(77, 237,4), (76, 235,4), (80, 235, 4)],
        lambda x,y,z: False,
        agent_space=False
    )
    experiment.room_11(
        bootstrap_env
    )
