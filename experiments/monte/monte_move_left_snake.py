from experiments.monte.environment import MonteBootstrapWrapper
from experiments import Experiment
import numpy as np
from pfrl.wrappers import atari_wrappers

from experiments.monte.environment import MonteAgentWrapper
from portable.utils import load_init_states
import argparse
from portable.utils.ale_utils import get_object_position, get_skull_position
from experiments import check_termination_correct_enemy, get_percent_completed_enemy
from portable.utils.utils import load_gin_configs

initiation_positive_files = [
    'resources/monte_images/room9_move_left_snake_initiation_positive.npy',
    'resources/monte_images/room9_move_left_snake_initiation_1_positive.npy',
]
initiation_negative_files = [
    'resources/monte_images/room9_move_left_snake_initiation_negative.npy',
    'resources/monte_images/room9_move_left_snake_initiation_1_negative.npy',
]
initiation_priority_negative_files = [
    'resources/monte_images/screen_death_1.npy',
    'resources/monte_images/screen_death_2.npy',
    'resources/monte_images/screen_death_3.npy',
    'resources/monte_images/screen_death_4.npy',
    'resources/monte_images/screen_death_5.npy'
]

termination_positive_files = [
    'resources/monte_images/room9_move_left_snake_termination_positive.npy',
]
termination_negative_files = [
    'resources/monte_images/room9_move_left_snake_termination_1_negative.npy',
    'resources/monte_images/room9_move_left_snake_termination_negative.npy',
]
termination_priority_negative_files = [
]

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

    return MonteAgentWrapper(env, agent_space=False)

initiation_state_files = [
    [
        'resources/monte_env_states/room9/enemy/right_of_left_snake.pkl',
        'resources/monte_env_states/room9/enemy/right_of_right_snake.pkl',
    ],[
        'resources/monte_env_states/room11/enemy/right_of_right_snake.pkl',
        'resources/monte_env_states/room11/enemy/right_of_right_snake_2.pkl',
        'resources/monte_env_states/room11/enemy/right_of_right_snake_3.pkl',
    ],[
        'resources/monte_env_states/room22/enemy/right_snake.pkl',
    ],[
        'resources/monte_env_states/room2/enemy/right_skull_0.pkl',
        'resources/monte_env_states/room2/enemy/right_skull_1.pkl',
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
        (42, 235, 9),
    ],[
        (97, 235, 11),
        (97, 235, 11),
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
    "room9", # 0
    "room11_left", 
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
    check_termination_correct_enemy,
    agent_space=False
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
        get_percentage_function=get_percent_completed_enemy,
        check_termination_true_function=check_termination_correct_enemy,
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
        100,
        use_agent_space=False
    )

    for y in range(len(initiation_state_files)):
        idx = order[y]
        experiment.run_trial(
            load_init_states(initiation_state_files[idx]),
            terminations[idx],
            100,
            eval=True,
            trial_name="{}_eval_after_bootstrap".format(room_names[idx]),
            use_agent_space=False
        )

    experiment.save()

    for x in range(1, len(initiation_state_files)):
        idx = order[x]
        instantiation_instances = experiment.run_trial(
            load_init_states(initiation_state_files[idx]),
            terminations[idx],
            2000,
            eval=False,
            trial_name="{}_train".format(room_names[idx]),
            use_agent_space=False
        )
        experiment.test_assimilate(
                load_init_states(initiation_state_files[0]),
                terminations[0],
                instantiation_instances,
                500,
                trial_name="{}_assimilate_test_".format(room_names[idx]),
                use_agent_space=False
            )
        for y in range(len(initiation_state_files)):
            idy = order[y]
            experiment.run_trial(
                load_init_states(initiation_state_files[idy]),
                terminations[idy],
                500,
                eval=True,
                trial_name="{}_eval_after_{}_train".format(room_names[idy], room_names[idx]),
                use_agent_space=False
            )
        
        experiment.save()