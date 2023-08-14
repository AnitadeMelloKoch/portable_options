from experiments.factored_minigrid.experiment.factored_minigrid_experiment import FactoredMinigridExperiment
from experiments.minigrid.doorkey.core.agents.rainbow import Rainbow
import os
import numpy as np 
from experiments.factored_minigrid.utils import environment_builder
import argparse 

from portable.utils.utils import load_gin_configs
import torch


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
    
    def create_agent(
            n_actions,
            gpu,
            n_input_channels,
            env_steps=1_000_000,
            lr=1e-4,
            sigma=0.5
        ):
        kwargs = dict(
            n_atoms=51, v_max=10., v_min=-10.,
            noisy_net_sigma=sigma, lr=lr, n_steps=3,
            betasteps=env_steps // 4,
            replay_start_size=1024, 
            replay_buffer_size=int(3e5),
            gpu=gpu, n_obs_channels=n_input_channels,
            use_custom_batch_states=False,
            epsilon_decay_steps=50000 # don't forget to change
        )
        return Rainbow(n_actions, **kwargs)
    
    def create_env(seed):
        return environment_builder('FactoredMiniGrid-DoorKey-8x8-v0', 
                                   seed=seed,
                                   max_steps=2000)
    
    experiment = FactoredMinigridExperiment(base_dir=args.base_dir,
                            random_seed=args.seed,
                            create_env_function=create_env,
                            create_agent_function=create_agent,
                            action_space=7
                            )
    
    experiment.train_test_envs()
    
    