import argparse 
from portable.utils.utils import load_gin_configs
import torch 
from experiments.divdis_monte.core.monte_divdis_policy_experiment import MonteDivDisOptionExperiment
from experiments.monte.environment import MonteBootstrapWrapper, MonteAgentWrapper
from pfrl.wrappers import atari_wrappers
from experiments.divdis_monte.experiment_files import *
from portable.utils import load_init_states
import numpy as np

init_states = [
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
]

terminations = [
    (20, 148, 1),
    (20, 148, 1),
    (20, 148, 1),
    (77, 192, 1),
    (77, 192, 1),
    (77, 192, 1),
    (77, 192, 1),
    (134,148, 1),
    (134,148, 1),
    (134,148, 1),
    (134,148, 1)
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
    
    def policy_phi(x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        x = (x/255.0).float()
        return x
    
    experiment = MonteDivDisOptionExperiment(base_dir=args.base_dir,
                                             seed=args.seed,
                                             policy_phi=policy_phi,
                                             option_type="divdis")
    
    env = atari_wrappers.wrap_deepmind(
        atari_wrappers.make_atari('MontezumaRevengeNoFrameskip-v4'),
        episode_life=True,
        clip_rewards=True,
        frame_stack=False
    )
    env.seed(args.seed)
    
    env = MonteAgentWrapper(env, agent_space=False, stack_observations=False)
    env = MonteBootstrapWrapper(env,
                                agent_space=False,
                                list_init_states=load_init_states(init_states),
                                check_true_termination=epsilon_ball_termination,
                                list_termination_points=terminations,
                                max_steps=int(2e4))
    
    experiment.add_data(true_list=monte_positive_files[0],
                        false_list=monte_negative_files[0],
                        unlabelled_list=monte_unlabelled_files[0])
    
    experiment.train_classifier(400)
    
    experiment.train_option(env=env,
                            env_seed=args.seed,
                            max_steps=1e6)



