from portable.environment import MonteBootstrapWrapper
import os
from experiments import PolicyExperiment
import numpy as np
from pfrl.wrappers import atari_wrappers

from portable.environment import MonteAgentWrapper
from portable.utils import load_init_states
import argparse

from portable.utils.utils import load_gin_configs

def get_percent_completed(start_pos, final_pos, terminations, env):
    top_of_room = 253
    bottom_of_room = 135

    def manhatten(top,bot):
        distance = 0
        # in same room
        if top[2] == bot[2]:
            distance = sum(abs(val1-val2) for val1, val2 in zip((top[0], top[1]),(bot[0],bot[1])))
        # not in same room
        else:
            distance += sum(abs(val1-val2) for val1, val2 in zip((top[0], top[1]),(top[0],bottom_of_room)))
            distance += sum(abs(val1-val2) for val1, val2 in zip((bot[0], bot[1]),(bot[0],top_of_room)))
        return distance

    true_distance = []
    completed_distance = []
    for term in terminations:
        true_distance.append(manhatten(start_pos, term))
        completed_distance.append(manhatten(start_pos, final_pos))
    true_distance = np.min(true_distance)+1e-5
    completed_distance = np.min(completed_distance)+1e-5

    return completed_distance/true_distance


def check_termination_correct(final_pos, terminations, env):
    epsilon = 2
    def in_epsilon_square(current_position, final_position):
        if current_position[0] <= (final_position[0] + epsilon) and \
            current_position[0] >= (final_position[0] - epsilon) and \
            current_position[1] <= (final_position[1] + epsilon) and \
            current_position[1] >= (final_position[1] - epsilon):
            return True
        return False 

    for term in terminations:
        if term[2] == final_pos[2]:
            if in_epsilon_square(final_pos, term):
                return True

    return False

def phi(x):
    return np.asarray(x, dtype=np.float32)

def make_env(seed):
    env = atari_wrappers.wrap_deepmind(
        atari_wrappers.make_atari('MontezumaRevengeNoFrameskip-v4', max_frames=1000),
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
        'resources/monte_env_states/room13/ladder/bottom_3.pkl',
        'resources/monte_env_states/room13/ladder/bottom_4.pkl',
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
    ],[
        [(77,235,11)],
        [(77,235,11)],
        [(77,235,11)],
        [(77,235,11)],
    ],[
        [(77,235,22)],
        [(77,235,22)],
        [(77,235,22)],
    ]
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

    experiment = PolicyExperiment(
        base_dir=args.base_dir,
        seed=args.seed,
        policy_phi=phi
    )

    for idx in range(len(initiation_state_files)):
        bootstrap_env = atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari('MontezumaRevengeNoFrameskip-v4', max_frames=1000),
            episode_life=True,
            clip_rewards=True,
            frame_stack=False
        )
        bootstrap_env = MonteBootstrapWrapper(
            bootstrap_env,
            load_init_states(initiation_state_files[idx]),
            terminations[idx],
            check_termination_correct,
            agent_space=True
        )

        experiment.run_trial(
            bootstrap_env,
            5000000
        )
