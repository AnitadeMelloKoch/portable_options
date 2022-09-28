from portable.environment import MonteBootstrapWrapper
from experiments import Experiment
import numpy as np
from pfrl.wrappers import atari_wrappers

from portable.environment import MonteAgentWrapper
from portable.utils import load_init_states
import argparse
from portable.utils.ale_utils import get_object_position, get_skull_position

from portable.utils.utils import load_gin_configs

# right snakes
def get_snake_x(position):
    if position[2] == 9:
        if position[1] <= 42:
            return 42
        else:
            return 11
    if position[2] == 11:
        if position[1] <= 97:
            return 97
        else:
            return 35
    if position[2] == 22:
        return 25

def in_epsilon_square(current_position, final_position):
    epsilon = 2
    if current_position[0] <= (final_position[0] + epsilon) and \
        current_position[0] >= (final_position[0] - epsilon) and \
        current_position[1] <= (final_position[1] + epsilon) and \
        current_position[1] >= (final_position[1] - epsilon):
        return True
    return False   

def get_percent_completed(start_pos, final_pos, terminations, env):
    def manhatten(a,b):
        return sum(abs(val1-val2) for val1, val2 in zip((a[0], a[1]),(b[0],b[1])))

    if start_pos[2] != final_pos[2]:
        return 0

    room = start_pos[2]
    ground_y = start_pos[1]
    ram = env.unwrapped.ale.getRAM()
    original_distance = 0
    final_distance = 0
    if room in [0,1,2,3,18]:
        # skulls
        skull_x = get_skull_position(ram)
        end_pos = (skull_x-6, ground_y)
        if final_pos[0] < skull_x and final_pos[1] <= ground_y:
            return 1
        else:
            original_distance = manhatten(start_pos, end_pos)
            final_distance = manhatten(final_pos, end_pos)
    elif room in [4,13,21]:
        # spiders
        spider_x, _ = get_object_position(ram)
        end_pos = (spider_x - 6, ground_y)
        if final_pos[0] < spider_x and final_pos[1] <= ground_y:
            return 1
        else:
            original_distance = manhatten(start_pos, end_pos)
            final_distance = manhatten(final_pos, end_pos)
    elif room in [9,11,22]:
        # snakes
        end_pos = terminations
        if in_epsilon_square(final_pos, end_pos):
            return 1
        else:
            original_distance = manhatten(start_pos, end_pos)
            final_distance = manhatten(final_pos, end_pos)
    else:
        return 0

    return 1 - final_distance/(original_distance+1e-5)

def check_termination_correct(final_pos, terminations, env):
    if terminations[2] != final_pos[2]:
        return 0

    room = terminations[2]
    ground_y = terminations[1]
    ram = env.unwrapped.ale.getRAM()
    if room in [0,1,2,3,18]:
        # skulls
        skull_x = get_skull_position(ram)
        end_pos = (skull_x-6, ground_y)
        if final_pos[0] < skull_x and final_pos[1] <= ground_y:
            return True
        else:
            return False
    elif room in [4,13,21]:
        # spiders
        spider_x, _ = get_object_position(ram)
        end_pos = (spider_x - 6, ground_y)
        if final_pos[0] < spider_x and final_pos[1] <= ground_y:
            return True
        else:
            return False
    elif room in [9,11,22]:
        # snakes
        test_pos = (final_pos[0]-3, final_pos[1], final_pos[2])
        snake_x = get_snake_x(test_pos)
        end_pos = (snake_x, ground_y)
        if final_pos[0] < snake_x and final_pos[1] <= ground_y:
            return True
        else:
            return False
    else:
        return False

initiation_positive_files = [
    'resources/monte_images/snake_1_initiation_positive.npy',
    'resources/monte_images/snake_2_initiation_positive.npy'
]
initiation_negative_files = [
    'resources/monte_images/snake_1_initiation_negative.npy',
    'resources/monte_images/snake_2_initiation_negative.npy',
    'resources/monte_images/jump_left_initiation_negative.npy',
    
]
initiation_priority_negative_files = [
    'resources/monte_images/death.npy',
    'resources/monte_images/falling_1.npy',
    'resources/monte_images/falling_2.npy',
    'resources/monte_images/falling_3.npy'
]

termination_positive_files = [
    'resources/monte_images/snake_1_termination_positive.npy',
    'resources/monte_images/snake_2_termination_positive.npy'
]
termination_negative_files = [
    'resources/monte_images/snake_1_termination_negative.npy',
    'resources/monte_images/snake_2_termination_negative.npy',
    'resources/monte_images/jump_left_termination_negative.npy',

]
termination_priority_negative_files = [
    'resources/monte_images/snake_1_termination_positive.npy',
    'resources/monte_images/snake_2_termination_positive.npy'
]

def phi(x):
    return np.asarray(x, dtype=np.float32)/255

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
        'resources/monte_env_states/room9/enemy/right_of_left_snake.pkl',
    ],[
        'resources/monte_env_states/room9/enemy/right_of_right_snake.pkl',
    ],[
        'resources/monte_env_states/room11/enemy/right_of_left_snake.pkl',
    ],[
        'resources/monte_env_states/room11/enemy/right_of_right_snake.pkl',
    ],[
        'resources/monte_env_states/room22/enemy/right_snake.pkl',
    ],[
        'resources/monte_env_states/room2/enemy/right_of_skull_0.pkl',
        'resources/monte_env_states/room2/enemy/right_of_skull_1.pkl',
    ],[
        'resources/monte_env_states/room4/enemy/right_of_spider_0.pkl',
        'resources/monte_env_states/room4/enemy/right_of_spider_1.pkl',
    ],[
        'resources/monte_env_states/room1/enemy/right_of_skull_0.pkl',
        'resources/monte_env_states/room1/enemy/right_of_skull_1.pkl',
    ]
]

terminations = [
    [
        (11, 235, 9),
    ],[
        (42, 235, 9),
    ],[
        (35, 235, 11),
    ],[
        (97, 235, 11),
    ],[
        (25, 235, 22)
    ],[
        (0, 235, 2),
        (0, 235, 2),
    ],[
        (0, 235, 4),
        (0, 235, 4),
    ],[
        (0, 148, 1),
        (0, 148, 1),
    ]
]

room_names = [
    "room9_left", # 0
    "room9_right", # 1
    "room11_left", # 2
    "room11_right",
    "room22",
    "room2",
    "room4",
    "room1",
]

order = [
    0, 1, 2, 3, 4, 5, 6, 7
]

bootstrap_env = atari_wrappers.wrap_deepmind(
        atari_wrappers.make_atari('MontezumaRevengeNoFrameskip-v4', max_frames=1000),
        episode_life=True,
        clip_rewards=True,
        frame_stack=False
    )
bootstrap_env = MonteBootstrapWrapper(
    bootstrap_env,
    load_init_states(initiation_state_files[0]),
    terminations[0],
    check_termination_correct,
    agent_space=True
)

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
        get_percentage_function=get_percent_completed,
        check_termination_true_function=check_termination_correct,
        policy_bootstrap_env=bootstrap_env,
        initiation_positive_files=initiation_positive_files,
        initiation_negative_files=initiation_negative_files,
        initiation_priority_negative_files=initiation_priority_negative_files,
        termination_positive_files=termination_positive_files,
        termination_negative_files=termination_negative_files,
        termination_priority_negative_files=termination_priority_negative_files
    )

    experiment.save()

    experiment.bootstrap_from_room(
        load_init_states(initiation_state_files[0]),
        terminations[0],
        50,
        use_agent_space=True
    )

    for y in range(len(initiation_state_files)):
        idx = order[y]
        experiment.run_trial(
            load_init_states(initiation_state_files[idx]),
            terminations[idx],
            50,
            eval=True,
            trial_name="{}_eval_after_bootstrap".format(room_names[idx]),
            use_agent_space=True
        )

    experiment.save()

    for x in range(1, len(initiation_state_files)):
        idx = order[x]
        experiment.run_trial(
            load_init_states(initiation_state_files[idx]),
            terminations[idx],
            100,
            eval=False,
            trial_name="{}_train".format(room_names[idx]),
            use_agent_space=True
        )
        for y in range(len(initiation_state_files)):
            idy = order[y]
            experiment.run_trial(
                load_init_states(initiation_state_files[idy]),
                terminations[idy],
                50,
                eval=True,
                trial_name="{}_eval_after_{}_train".format(room_names[idy], room_names[idx]),
                use_agent_space=True
            )
        
        experiment.save()

    
    

