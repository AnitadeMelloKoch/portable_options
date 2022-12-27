from portable.environment import MonteBootstrapWrapper
from experiments import Experiment
import numpy as np
from pfrl.wrappers import atari_wrappers

from portable.environment import MonteAgentWrapper
from portable.utils import load_init_states
import argparse
from portable.utils.ale_utils import get_object_position, get_skull_position
from experiments import check_termination_correct_enemy, get_percent_completed_enemy
from portable.utils.utils import load_gin_configs

initiation_positive_files = [
    'resources/monte_images/rolling_skull_1_initiation_positive.npy',
    'resources/monte_images/rolling_skull_2_initiation_positive.npy'
]
initiation_negative_files = [
    'resources/monte_images/rolling_skull_1_initiation_negative.npy',
    'resources/monte_images/rolling_skull_2_initiation_negative.npy',
    'resources/monte_images/jump_left_initiation_negative.npy',
    
]
initiation_priority_negative_files = [
    'resources/monte_images/death.npy',
    'resources/monte_images/falling_1.npy',
    'resources/monte_images/falling_2.npy',
    'resources/monte_images/falling_3.npy'
]

termination_positive_files = [
    'resources/monte_images/rolling_skull_1_termination_positive.npy',
    'resources/monte_images/rolling_skull_2_termination_positive.npy'
]
termination_negative_files = [
    'resources/monte_images/rolling_skull_1_termination_negative.npy',
    'resources/monte_images/rolling_skull_2_termination_negative.npy',
    'resources/monte_images/jump_left_termination_negative.npy',

]
termination_priority_negative_files = [
    'resources/monte_images/rolling_skull_1_termination_positive.npy',
    'resources/monte_images/rolling_skull_2_termination_positive.npy'
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
        'resources/monte_env_states/room5/enemy/right_of_skull.pkl',
    ],[
        'resources/monte_env_states/room18/enemy/right_skull.pkl',
    ],[
        'resources/monte_env_states/room4/enemy/right_of_spider_0.pkl',
        'resources/monte_env_states/room4/enemy/right_of_spider_1.pkl',
    ],[
        'resources/monte_env_states/room2/enemy/right_of_skull_0.pkl',
        'resources/monte_env_states/room2/enemy/right_of_skull_1.pkl',
    ],[
        'resources/monte_env_states/room11/enemy/right_of_right_snake.pkl',
    ]
]

terminations = [
    [
        (0, 148, 1),
        (0, 148, 1),
    ],[
        (0, 235, 5),
    ],[
        (0, 235, 18),
    ],[
        (0, 235, 4),
        (0, 235, 4),
    ],[
        (0, 235, 2),
        (0, 235, 2),
    ],[
        (97, 235, 11)
    ]
]

room_names = [
    "room1", # 0
    "room5", # 1
    "room18", # 2
    "room4",
    "room2",
    "room11",
]

order = [
    0, 1, 2, 3, 4, 5
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
            50,
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

    
    

