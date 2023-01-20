from portable.environment import MonteBootstrapWrapper, MonteAgentWrapper
from experiments import RainbowExperiment
from pfrl.wrappers import atari_wrappers
from portable.utils import load_init_states
from portable.utils import get_skull_position
from experiments import check_termination_correct_enemy
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

def check_right_skull(final_pos, terminations, env):
    if terminations[2] != final_pos[2]:
        return 0
    room = terminations[2]
    ground_y = terminations[1]
    ram = env.unwrapped.ale.getRAM()
    if final_pos[2] != room:
        return False

    if room in [0,1,18]:
            # skulls
            skull_x = get_skull_position(ram)
            end_pos = (skull_x+6, ground_y)
            if final_pos[0] < skull_x+6 and final_pos[1] <= ground_y:
                return True
            else:
                return False

def check_termination_correct_from_jump_right(final_pos, terminations, env): 
    # TERMINATIONS HERE IS THE STARTING POSITION
    info = env.get_current_info({})

    if terminations[2] == final_pos[2]:
        if env.jumped_previously is True and info["falling"] is 0 and final_pos[0] > terminations[0]:
            return True

    return False

def check_termination_correct_from_jump_left(final_pos, terminations, env): 
    # TERMINATIONS HERE IS THE STARTING POSITION
    info = env.get_current_info({})

    if terminations[2] == final_pos[2]:
        if env.jumped_previously is True and info["falling"] is 0 and final_pos[0] < terminations[0]:
            return True

    return False

def check_termination_correct_from_jump_up(final_pos, terminations, env): 
    # TERMINATIONS HERE IS THE STARTING POSITION
    info = env.get_current_info({})

    if terminations[2] == final_pos[2]:
        if env.jumped_previously is True and info["falling"] is 0 and \
                final_pos[0] > (terminations[0]-4) and final_pos[0] < (terminations[0]+4):
            return True

    return False

def true_initiation_climb_down_ladder(position):
    if position[2] != 1:
        return False
    if position[1] in [21, 20, 22, 133, 134, 135]:
        if position[0] < 195 and position[0] > 147:
            return True
    if position[1] in [76, 77, 78]:
        if position[0] < 238 and position[0] > 193:
            return True
    return False

def true_initiation_climb_up_ladder(position):
    if position[2] != 1:
        return False
    if position[1] in [21, 20, 22, 133, 134, 135]:
        if position[0] < 191 and position[0] > 144:
            return True
    if position[1] in [76, 77, 78]:
        if position[0] < 234 and position[0] > 190:
            return True
    return False

def true_initiation_rope(position):
    if position[2] != 1:
        return False
    if position[0] == 109:
        if position[1] > 180 and position[1] < 214:
            return True
    return False

def true_initiation_jump_left(position):
    if position in [(69, 235, 1),
        (104, 235, 1),
        (130, 192, 1),
        (109, 201, 1),
        (109, 209, 1)]:
        return True
    return False

def true_initiation_jump_right(position):
    if position in [(50, 235, 1),
        (92, 192, 1),
        (85, 235, 1),
        (109, 201, 1),
        (109, 209, 1)]:
        return True
    return False

def true_initiation_jump_up(position):
    if position == (17, 192, 1):
        return True
    return False

def true_initiation_jump_skull(position):
    # not amazing but close enough?
    if position[2] != 1:
        return False
    if position[1] == 148:
        return True

true_init_functions = [
    true_initiation_climb_down_ladder,
    true_initiation_climb_up_ladder,
    true_initiation_rope,
    true_initiation_rope,
    true_initiation_jump_left,
    true_initiation_jump_right,
    true_initiation_jump_up,
    true_initiation_jump_skull,
    true_initiation_jump_skull
]

def phi(x):
    return np.asarray(x, dtype=np.float32) / 255

def make_bootstrap_env():
    return atari_wrappers.wrap_deepmind(
        atari_wrappers.make_atari('MontezumaRevengeNoFrameskip-v4', max_frames=1000),
        episode_life=True,
        clip_rewards=True,
        frame_stack=False
    )

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
        'resources/monte_images/climb_down_ladder_initiation_positive.npy'
    ],[
        'resources/monte_images/climb_up_ladder_initiation_positive.npy'
    ],[
        'resources/monte_images/climb_down_rope_initiation_positive.npy'
    ],[
        'resources/monte_images/climb_up_rope_initiation_positive.npy'
    ],[
        'resources/monte_images/jump_left_initiation_positive.npy'
    ],[
        'resources/monte_images/jump_right_initiation_positive.npy'
    ],[
        'resources/monte_images/jump_up_initiation_positive.npy'
    ],[
        'resources/monte_images/rolling_skull_1_initiation_positive.npy',
        'resources/monte_images/rolling_skull_2_initiation_positive.npy'
    ],[
        'resources/monte_images/rolling_skull_1_termination_positive.npy',
        'resources/monte_images/rolling_skull_2_termination_positive.npy'
    ]
]

initiation_negative_files = [
    [
        'resources/monte_images/climb_down_ladder_initiation_negative.npy'
    ],[
        'resources/monte_images/climb_up_ladder_initiation_negative.npy'
    ],[
        'resources/monte_images/climb_down_rope_initiation_negative.npy'
    ],[
        'resources/monte_images/climb_up_rope_initiation_negative.npy'
    ],[
        'resources/monte_images/jump_left_initiation_negative.npy'
    ],[
        'resources/monte_images/jump_right_initiation_negative.npy'
    ],[
        'resources/monte_images/jump_up_initiation_negative.npy'
    ],[
        'resources/monte_images/rolling_skull_1_initiation_negative.npy',
        'resources/monte_images/rolling_skull_2_initiation_negative.npy',
        'resources/monte_images/jump_left_initiation_negative.npy',
    ],[
        'resources/monte_images/rolling_skull_1_termination_negative.npy',
        'resources/monte_images/jump_left_initiation_negative.npy',
        'resources/monte_images/rolling_skull_2_termination_negative.npy',
    ]
]

initiation_priority_negative_files = [
    [
        'resources/monte_images/death.npy',
        'resources/monte_images/falling_1.npy',
        'resources/monte_images/falling_2.npy',
        'resources/monte_images/falling_3.npy',
        'resources/monte_images/climb_down_ladder_1_termination_positive.npy',
        'resources/monte_images/climb_down_ladder_2_termination_positive.npy',
        'resources/monte_images/climb_down_ladder_3_termination_positive.npy',
        'resources/monte_images/climb_down_ladder_4_termination_positive.npy',
    ],[
        'resources/monte_images/climb_up_ladder_termination_positive.npy',
        'resources/monte_images/death.npy',
        'resources/monte_images/falling_1.npy',
        'resources/monte_images/falling_2.npy',
        'resources/monte_images/falling_3.npy',
    ],[
        'resources/monte_images/death.npy',
        'resources/monte_images/falling_1.npy',
        'resources/monte_images/falling_2.npy',
        'resources/monte_images/falling_3.npy',
    ],[
        'resources/monte_images/death.npy',
        'resources/monte_images/falling_1.npy',
        'resources/monte_images/falling_2.npy',
        'resources/monte_images/falling_3.npy',
    ],[
        'resources/monte_images/death.npy',
    ],[
        'resources/monte_images/death.npy',
    ],[
        'resources/monte_images/death.npy',
    ],[
        'resources/monte_images/death.npy',
        'resources/monte_images/falling_1.npy',
        'resources/monte_images/falling_2.npy',
        'resources/monte_images/falling_3.npy'
    ],[
        'resources/monte_images/death.npy',
        'resources/monte_images/falling_1.npy',
        'resources/monte_images/falling_2.npy',
        'resources/monte_images/falling_3.npy'
    ]
]

termination_positive_files = [
    [
        'resources/monte_images/climb_down_ladder_1_termination_positive.npy',
        'resources/monte_images/climb_down_ladder_2_termination_positive.npy',
        'resources/monte_images/climb_down_ladder_3_termination_positive.npy',
        'resources/monte_images/climb_down_ladder_4_termination_positive.npy',
    ],[
        'resources/monte_images/climb_up_ladder_termination_positive.npy'
    ],[
        'resources/monte_images/climb_down_rope_termination_positive.npy'
    ],[
        'resources/monte_images/climb_up_rope_termination_positive.npy'
    ],[
        'resources/monte_images/jump_left_termination_positive.npy'
    ],[
        'resources/monte_images/jump_right_termination_positive.npy'
    ],[
        'resources/monte_images/jump_up_termination_positive.npy'
    ],[
        'resources/monte_images/rolling_skull_1_termination_positive.npy',
        'resources/monte_images/rolling_skull_2_termination_positive.npy'
    ],[
        'resources/monte_images/rolling_skull_1_initiation_positive.npy',
        'resources/monte_images/rolling_skull_2_initiation_positive.npy'
    ]
]

termination_negative_files = [
    [
        'resources/monte_images/climb_down_ladder_termination_negative.npy'
    ],[
        'resources/monte_images/climb_up_ladder_termination_negative.npy'
    ],[
        'resources/monte_images/climb_down_rope_termination_negative.npy'
    ],[
        'resources/monte_images/climb_up_rope_termination_negative.npy'
    ],[
        'resources/monte_images/jump_left_termination_negative.npy'
    ],[
        'resources/monte_images/jump_right_termination_negative.npy'
    ],[
        'resources/monte_images/jump_up_termination_negative.npy'
    ],[
        'resources/monte_images/rolling_skull_1_termination_negative.npy',
        'resources/monte_images/rolling_skull_2_termination_negative.npy',
    ],[
        'resources/monte_images/rolling_skull_1_initiation_negative.npy',
        'resources/monte_images/rolling_skull_2_initiation_negative.npy',
    ]
]

termination_priority_negative_files = [
    [
        'resources/monte_images/climb_down_ladder_initiation_positive.npy'
    ],[
        'resources/monte_images/climb_up_ladder_initiation_positive.npy'
    ],[
    ],[
    ],[
        'resources/monte_images/jump_left_initiation_positive.npy'
    ],[
        'resources/monte_images/jump_right_initiation_positive.npy'
    ],[
        'resources/monte_images/jump_up_initiation_positive.npy'
    ],[
        'resources/monte_images/rolling_skull_1_initiation_positive.npy',
        'resources/monte_images/rolling_skull_2_initiation_positive.npy'
    ],[]
]

environment_rams = [
    [
        'resources/monte_env_states/room1/ladder/left_top_0.pkl',
        'resources/monte_env_states/room1/ladder/left_top_1.pkl',
        'resources/monte_env_states/room1/ladder/left_top_0.pkl',
        'resources/monte_env_states/room1/ladder/middle_top_0.pkl',
        'resources/monte_env_states/room1/ladder/middle_top_1.pkl',
        'resources/monte_env_states/room1/ladder/middle_top_2.pkl',
        'resources/monte_env_states/room1/ladder/middle_top_3.pkl',
        'resources/monte_env_states/room1/ladder/right_top_0.pkl',
        'resources/monte_env_states/room1/ladder/right_top_1.pkl',
        'resources/monte_env_states/room1/ladder/right_top_2.pkl',
        'resources/monte_env_states/room1/ladder/right_top_3.pkl',
    ],[
        'resources/monte_env_states/room1/ladder/left_bottom_0.pkl',
        'resources/monte_env_states/room1/ladder/left_bottom_1.pkl',
        'resources/monte_env_states/room1/ladder/left_bottom_2.pkl',
        'resources/monte_env_states/room1/ladder/middle_bottom_0.pkl',
        'resources/monte_env_states/room1/ladder/middle_bottom_1.pkl',
        'resources/monte_env_states/room1/ladder/middle_bottom_2.pkl',
        'resources/monte_env_states/room1/ladder/right_bottom_0.pkl',
        'resources/monte_env_states/room1/ladder/right_bottom_1.pkl',
        'resources/monte_env_states/room1/ladder/right_bottom_2.pkl',
    ],[
        'resources/monte_env_states/room1/rope/rope_mid_1.pkl',
        'resources/monte_env_states/room1/rope/rope_mid_2.pkl',
        'resources/monte_env_states/room1/rope/rope_top_1.pkl',
        'resources/monte_env_states/room1/rope/rope_top_2.pkl',
    ],[
        'resources/monte_env_states/room1/rope/rope_bot_1.pkl',
        'resources/monte_env_states/room1/rope/rope_bot_2.pkl',
        'resources/monte_env_states/room1/rope/rope_mid_1.pkl',
        'resources/monte_env_states/room1/rope/rope_mid_2.pkl',
        'resources/monte_env_states/room1/rope/rope_mid_2.pkl',
        'resources/monte_env_states/room1/rope/rope_top_1.pkl',
    ],[
        'resources/monte_env_states/room1/platforms/middle_ladder_top_left.pkl',
        'resources/monte_env_states/room1/platforms/right_top_platform_left.pkl',
        'resources/monte_env_states/room1/platforms/middle_right_platform_left.pkl',
        'resources/monte_env_states/room1/rope/rope_mid_2.pkl',
        'resources/monte_env_states/room1/rope/rope_top_1.pkl',
    ],[
        'resources/monte_env_states/room1/platforms/left_top_platform_right.pkl',
        'resources/monte_env_states/room1/platforms/middle_ladder_bottom_right.pkl',
        'resources/monte_env_states/room1/platforms/middle_ladder_top_right.pkl',
        'resources/monte_env_states/room1/rope/rope_mid_2.pkl',
        'resources/monte_env_states/room1/rope/rope_top_1.pkl',
    ],[
        'resources/monte_env_states/room1/platforms/under_key_1.pkl',
        'resources/monte_env_states/room1/platforms/under_key_2.pkl',
        'resources/monte_env_states/room1/platforms/under_key_3.pkl',
    ],[
        'resources/monte_env_states/room1/enemy/right_of_skull_0.pkl',
        'resources/monte_env_states/room1/enemy/right_of_skull_1.pkl',
    ],[
        'resources/monte_env_states/room1/enemy/left_of_skull_0.pkl',
        'resources/monte_env_states/room1/enemy/left_of_skull_1.pkl'
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
        [(21,192,1),(20,192,1),(22,192,1)],
        [(21,192,1),(20,192,1),(22,192,1)],
        [(21,192,1),(20,192,1),(22,192,1)],
        [(77,235,1)],
        [(77,235,1)],
        [(77,235,1)],
        [(129,192,1)],
        [(129,192,1)],
        [(129,192,1)]
    ],[
        [(109, 189, 1)],
        [(109, 189, 1)],
        [(109, 189, 1)],
        [(109, 189, 1)]
    ],[
        [(109, 209, 1)],
        [(109, 209, 1)],
        [(109, 209, 1)],
        [(109, 209, 1)],
        [(109, 209, 1)],
        [(109, 209, 1)]
    ],[
        (69, 235, 1),
        (104, 235, 1),
        (130, 192, 1),
        (109, 201, 1),
        (109, 209, 1)
    ],[
        (50, 235, 1),
        (92, 192, 1),
        (85, 235, 1),
        (109, 201, 1),
        (109, 209, 1)
    ],[
        (17, 192, 1),
        (17, 192, 1),
        (17, 192, 1)
    ],[
        (0, 148, 1),
        (0, 148, 1),
    ],[
        (0, 148, 1),
        (0, 148, 1),
    ]
]

bootstrap_envs = [
    MonteBootstrapWrapper(
        make_bootstrap_env(),
        load_init_states(environment_rams[0]),
        terminations[0],
        check_termination_correct_from_position,
        agent_space=True
    ),MonteBootstrapWrapper(
        make_bootstrap_env(),
        load_init_states(environment_rams[1]),
        terminations[1],
        check_termination_correct_from_position,
        agent_space=True
    ),MonteBootstrapWrapper(
        make_bootstrap_env(),
        load_init_states(environment_rams[2]),
        terminations[2],
        check_termination_correct_from_position,
        agent_space=True
    ),MonteBootstrapWrapper(
        make_bootstrap_env(),
        load_init_states(environment_rams[3]),
        terminations[3],
        check_termination_correct_from_position,
        agent_space=True
    ),MonteBootstrapWrapper(
        make_bootstrap_env(),
        load_init_states(environment_rams[4]),
        terminations[4],
        check_termination_correct_from_jump_left,
        agent_space=True
    ),MonteBootstrapWrapper(
        make_bootstrap_env(),
        load_init_states(environment_rams[5]),
        terminations[5],
        check_termination_correct_from_jump_right,
        agent_space=True
    ),MonteBootstrapWrapper(
        make_bootstrap_env(),
        load_init_states(environment_rams[6]),
        terminations[6],
        check_termination_correct_from_jump_up,
        agent_space=True
    ),MonteBootstrapWrapper(
        make_bootstrap_env(),
        load_init_states(environment_rams[7]),
        terminations[7],
        check_termination_correct_enemy,
        agent_space=True
    ),MonteBootstrapWrapper(
        make_bootstrap_env(),
        load_init_states(environment_rams[8]),
        terminations[8],
        check_right_skull,
        agent_space=True
    )
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

    experiment = RainbowExperiment(
        base_dir=args.base_dir,
        seed=args.seed,
        starting_action_num=len(environment_rams),
        experiment_env_function=make_env,
        policy_phi=phi,
        options_initiation_positive_files=initiation_positive_files,
        options_initiation_negative_files=initiation_negative_files,
        options_initiation_priority_negative_files=initiation_priority_negative_files,
        options_termination_positive_files=termination_positive_files,
        options_termination_negative_files=termination_negative_files,
        options_termination_priority_negative_files=termination_priority_negative_files,
        policy_bootstrap_envs=bootstrap_envs,
        true_init_functions=true_init_functions
    )

    experiment.run_trial(1000000)
