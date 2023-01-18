from portable.environment import MonteBootstrapWrapper, MonteAgentWrapper
from experiments import RainbowExperiment
from pfrl.wrappers import atari_wrappers
from portable.utils import load_init_states
import argparse
import numpy as np

from portable.utils.utils import load_gin_configs

def check_termination_correct_from_position(final_pos, terminations, env):
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
    return np.asarray(x, dtype=np.float32) / 255

def make_env(seed):
    env = atari_wrappers.wrap_deepmind(
        atari_wrappers.make_atari('MontezumaRevengeNoFrameskip-v4', max_frames=1000),
        episode_life=True,
        clip_rewards=True,
        frame_stack=False
    )
    env.seed(seed)

    return MonteAgentWrapper(env, agent_space=True)

# options
# climb down ladder
# climb up ladder
# climb down rope
# climb up rope
# jump left
# jump right
# jump up
# skull

initiation_positive_files = [
    [
        'resources/climb_down_ladder_initiation_positive.npy'
    ],[
        'resources/climb_up_ladder_initiation_positive.npy'
    ],[
        'resources/climb_down_rope_initiation_positive.npy'
    ],[
        'resources/climb_up_rope_initiation_positive.npy'
    ],[
        'resources/jump_left_initiation_positive.npy'
    ],[
        'resources/jump_right_initiation_positive.npy'
    ],[
        'resources/jump_up_initiation_positive.npy'
    ],[
        ''
    ]
]

initiation_negative_files = [
    [
        'resources/climb_down_ladder_initiation_negative.npy'
    ],[
        'resources/climb_up_ladder_initiation_negative.npy'
    ],[
        'resources/climb_down_rope_initiation_negative.npy'
    ],[
        'resources/climb_up_rope_initiation_negative.npy'
    ],[
        'resources/jump_left_initiation_negative.npy'
    ],[
        'resources/jump_right_initiation_negative.npy'
    ],[
        'resources/jump_up_initiation_negative.npy'
    ]
]

initiation_priority_negative_files = []

termination_positive_files = [
    [
        'resources/climb_down_ladder_1_termination_positive.npy',
        'resources/climb_down_ladder_2_termination_positive.npy',
        'resources/climb_down_ladder_3_termination_positive.npy',
        'resources/climb_down_ladder_4_termination_positive.npy',
    ],[
        'resources/climb_up_ladder_termination_positive.npy'
    ],[
        'resources/climb_down_rope_termination_positive.npy'
    ],[
        'resources/climb_up_rope_initiation_positive.npy'
    ],[
        'resources/jump_left_initiation_positive.npy'
    ],[
        'resources/jump_right_initiation_positive.npy'
    ],[
        'resources/jump_up_initiation_positive.npy'
    ]
]

termination_negative_files = [
    [
        'resources/climb_down_ladder_initiation_negative.npy'
    ],[
        'resources/climb_up_ladder_initiation_negative.npy'
    ],[
        'resources/climb_down_rope_initiation_negative.npy'
    ],[
        'resources/climb_up_rope_initiation_negative.npy'
    ],[
        'resources/jump_left_initiation_negative.npy'
    ],[
        'resources/jump_right_initiation_negative.npy'
    ],[
        'resources/jump_up_initiation_negative.npy'
    ]
]

termination_priority_neagtive_files = []

environment_rams = [
    []
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


