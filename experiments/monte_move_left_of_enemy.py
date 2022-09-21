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
room_to_snake_x = {
    9: 51,
    11: 108,
    22: 31,
}

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
    if room in [0,2,3,18]:
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
        snake_x = room_to_snake_x[room]
        end_pos = (snake_x-6, ground_y)
        if final_pos[0] < snake_x and final_pos[1] <= ground_y:
            return 1
        else:
            original_distance = manhatten(start_pos, end_pos)
            final_distance = manhatten(final_pos, end_pos)
    else:
        return 0

    return final_distance/original_distance

def check_termination_correct(final_pos, terminations, env):
    if terminations[2] != final_pos[2]:
        return 0

    room = terminations[2]
    ground_y = terminations[1]
    ram = env.unwrapped.ale.getRAM()
    if room in [0,2,3,18]:
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
        snake_x = room_to_snake_x[room]
        end_pos = (snake_x-6, ground_y)
        if final_pos[0] < snake_x and final_pos[1] <= ground_y:
            return True
        else:
            return False
    else:
        return False

initiation_positive_files = [
    'resources/monte_images/jump_left_initiation_positive.npy'
]
initiation_negative_files = [
    'resources/monte_images/jump_left_initiation_negative.npy'
]
initiation_priority_negative_files = [
    'resources/monte_images/death.npy',
    'resources/monte_images/falling_1.npy',
    'resources/monte_images/falling_2.npy',
    'resources/monte_images/falling_3.npy'
]

termination_positive_files = [
    'resources/monte_images/jump_left_termination_positive.npy'
]
termination_negative_files = [
    'resources/monte_images/jump_left_termination_negative.npy'
]
termination_priority_negative_files = [
    'resources/monte_images/jump_left_initiation_positive.npy'
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
        'resources/monte_env_states/room1/enemy/right_of_skull_0.pkl',
        'resources/monte_env_states/room1/enemy/right_of_skull_1.pkl',
    ],[
        'resources/monte_env_states/room2/enemy/right_of_skull_0.pkl',
        'resources/monte_env_states/room2/enemy/right_of_skull_1.pkl',
    ],[
        'resources/monte_env_states/room4/enemy/right_of_spider_0.pkl',
        'resources/monte_env_states/room4/enemy/right_of_spider_1.pkl',
    ],[
        'resources/monte_env_states/room3/enemy/right_of_skulls.pkl',
    ],[
        'resources/monte_env_states/room9/enemy/right_of_right_snake.pkl',
    ],[
        'resources/monte_env_states/room11/enemy/right_of_right_snake.pkl',
    ],[
        'resources/monte_env_states/room13/enemy/right_spider.pkl',
    ],[
        'resources/monte_env_states/room18/enemy/right_skull.pkl',
    ],[
        'resources/monte_env_states/room21/enemy/right_spider.pkl',
    ],[
        'resources/monte_env_states/room22/enemy/right_snake.pkl',
    ]
]

terminations = [
    [
        (0, 148, 1),
        (0, 148, 1),
    ],[
        (0, 235, 2),
        (0, 235, 2)
    ],[
        (0, 235, 4),
        (0, 235, 4)
    ],[
        (0, 235, 3)
    ],[
        (0, 235, 9)
    ],[
        (0, 235, 11)
    ],[
        (0, 235, 13)
    ],[
        (0, 235, 18)
    ],[
        (0, 235, 21)
    ],[
        (0, 235, 22)
    ]
]

room_names = [
    "room1", # 0
    "room2", # 1
    "room4", # 2
    "room3", # 3
    "room9", # 4
    "room11",# 5
    "room13",# 6
    "room18",# 7
    "room21",# 8
    "room22",# 9
]

order = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9
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

    experiment.save(additional_path=room_names[0])

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
        
        experiment.save(additional_path=room_names[idx])

    
    

