from experiments.core.divdis_meta_experiment import DivDisMetaExperiment
import argparse
from portable.utils.utils import load_gin_configs
from portable.agent.model.ppo import create_atari_model
import numpy as np
import torch

from experiments.monte.environment import MonteBootstrapWrapper, MonteAgentWrapper
from pfrl.wrappers import atari_wrappers
from experiments.divdis_monte.core.monte_terminations import *

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
    
    def option_agent_phi(x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        x = (x/255.0).float()
        return x
    # monte has 18 actions
    
    terminations = [
        [check_termination_bottom_ladder],
        [check_termination_top_ladder],
        [check_termination_correct_enemy_left],
        [check_termination_correct_enemy_right]
    ]
    
    experiment = DivDisMetaExperiment(base_dir=args.base_dir,
                                      seed=args.seed,
                                      option_policy_phi=policy_phi,
                                      agent_phi=option_agent_phi,
                                      action_model=create_atari_model(4, 5),
                                      option_type="mock",
                                      terminations=terminations)
    
    env = atari_wrappers.wrap_deepmind(
        atari_wrappers.make_atari('MontezumaRevengeNoFrameskip-v4', max_frames=1000),
        episode_life=True,
        clip_rewards=True,
        frame_stack=False
    )
    env.seed(args.seed)

    env = MonteAgentWrapper(env, agent_space=False)
    
    experiment.train_meta_agent(env,
                                args.seed,
                                4e6,
                                0.9)