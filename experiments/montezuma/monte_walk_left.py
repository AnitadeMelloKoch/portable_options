from portable.environment import MonteBootstrapWrapper
from experiments import Experiment
import numpy as np
from pfrl.wrappers import atari_wrappers
import argparse

from portable.environment import MonteAgentWrapper
from portable.utils import load_gin_configs
from portable.utils.utils import load_init_states

def get_percent_completed(start_pos, final_pos, terminations, env):
    def manhatten(a, b):
        return sum(abs(val1-val2) for val1, val2 in zip((a[0], a[1]),(b[0],b[1])))

    original_distance = []
    completed_distance = []
    for term in terminations:
        original_distance.append(manhatten(start_pos, term))
        completed_distance.append(manhatten(final_pos, term))
    original_distance = np.mean(original_distance)+1e-5
    completed_distance = np.mean(completed_distance)+1e-5

    return 1 - (completed_distance/original_distance)

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

initiation_positive_files = [
    'resources/monte_images/walk_left_initiation_positive.npy'
]
initiation_negative_files = [
    'resources/monte_images/walk_left_initiation_negative.npy'
]
initiation_priority_negative_files = [
    'resources/monte_images/death.npy',
    'resources/monte_images/falling_1.npy',
    'resources/monte_images/falling_2.npy',
    'resources/monte_images/falling_3.npy'
]

termination_positive_files = [
    'resources/monte_images/walk_left_termination_positive.npy'
]
termination_negative_files = [
    'resources/monte_images/walk_left_termination_negative.npy'
]
termination_priority_negative_files = [
    'resources/monte_images/walk_left_initiation_positive.npy'
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
        'resources/monte_env_states/room1/platforms/left_top_platform_right.pkl',
        'resources/monte_env_states/room1/platforms/middle_right_platform_right.pkl',
        'resources/monte_env_states/room1/platforms/middle_left_platform_right.pkl',
        'resources/monte_env_states/room1/platforms/middle_ladder_top_right.pkl',
        'resources/monte_env_states/room1/platforms/middle_ladder_bottom_right.pkl',
        'resources/monte_env_states/room1/ladder/middle_top_0.pkl',
        'resources/monte_env_states/room1/ladder/left_top_0.pkl',
    ],[
        'resources/monte_env_states/room0/lasers/right_of_middle_lasers.pkl'
    ],[
        # maybe shouldn't use room 2
        'resources/monte_env_states/room2/ladder/top_0.pkl',
        'resources/monte_env_states/room2/platforms/right.pkl',
    ],[
        # maybe shouldn't use room 3
        'resources/monte_env_states/room3/ladder/top_0.pkl',
        'resources/monte_env_states/room3/platforms/right.pkl',
    ],[
        'resources/monte_env_states/room5/platforms/right.pkl',
        'resources/monte_env_states/room5/platforms/right_door.pkl',
        'resources/monte_env_states/room5/platforms/top_platform_middle_right.pkl',
        'resources/monte_env_states/room5/platforms/top_platform_middle_middle.pkl'
    ],[
        'resources/monte_env_states/room6/platforms/right.pkl',
        'resources/monte_env_states/room6/ladder/bottom_0.pkl',
    ],[
        'resources/monte_env_states/room7/lasers/middle_room_near_right_lasers.pkl',
        'resources/monte_env_states/room7/ladder/top_0.pkl',
    ],[
        'resources/monte_env_states/room8/platforms/bottom_right.pkl',
        'resources/monte_env_states/room8/platforms/bottom_middle.pkl',
        'resources/monte_env_states/room8/platforms/right.pkl',
    ],[
        'resources/monte_env_states/room9/platforms/right.pkl',
    ],[
        'resources/monte_env_states/room10/ladder/bottom_0.pkl',
    ],[
        'resources/monte_env_states/room14/ladder/top_0.pkl',
        'resources/monte_env_states/room14/platforms/bottom_right.pkl'
    ],[
        'resources/monte_env_states/room19/ladder/top_0.pkl',
        'resources/monte_env_states/room19/platforms/right.pkl',
    ]
]

terminations = [
    [
        [(24, 235, 1)],
        [(130, 192, 1)],
        [(9, 192, 1)],
        [(77, 235, 1)],
        [(74, 192, 1)],
        [(69, 235, 1)],
        [(9, 192, 1)]
    ],[
        [(146, 235, 0)]
    ],[
        [(7, 235, 2)],
        [(79, 235, 2)]
    ],[
        [(5, 235, 3)],
        [(76, 235, 3)]
    ],[
        [(77, 157, 5)],
        [(77, 157, 5)],
        [(59, 235, 5)],
        [(59, 235, 5)],
    ],[
        [(77,235,6)],
        [(5, 235, 6)]
    ],[
        [(79, 235, 7)],
        [(51, 235, 7)]
    ],[
        [(13, 155, 8)],
        [(13, 155, 8)],
        [(120, 235, 8)]
    ],[
        [(77,235,9)]
    ],[
        [(5, 235, 10)]
    ],[
        [(32,160,14)],
        [(80,160,14)]
    ],[
        [(5,235,19)],
        [(77,235,19)]
    ]
]

room_names = [
    'room1', # 0
    'room0', # 1
    'room2', # 2
    'room3', # 3
    'room5', # 4
    'room6', # 5
    'room7', # 6
    'room8', # 7
    'room9', # 8
    'room10',# 9
    'room14',# 10
    'room19',# 11
]

order = [
    0 ,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
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

    # experiment.save()

    experiment.bootstrap_from_room(
        load_init_states(initiation_state_files[0]),
        terminations[0],
        50,
        use_agent_space=True
    )

    for y in range(len(initiation_state_files)):
        idx = order[y]
        experiment.run_trial(
            load_init_states(initiation_state_files[y]),
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

    

