from experiments.core.divdis_meta_masked_ppo_experiment import DivDisMetaMaskedPPOExperiment
import argparse
from portable.utils.utils import load_gin_configs, load_init_states
from experiments.minigrid.utils import environment_builder
from experiments.minigrid.advanced_doorkey.core.policy_train_wrapper import CorridorWrapper
import matplotlib.pyplot as plt
import random
import numpy as np
import os
import torch
from portable.agent.model.maskable_ppo import create_mask_cnn_policy
from portable.agent.model.ppo import create_cnn_vf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mask", action='store_true')
    parser.add_argument("--config_file", nargs='+', type=str, required=True)
    parser.add_argument("--gin_bindings", default=[], help='Gin bindings to override the values' + 
            ' set in the config files (e.g. "DQNAgent.epsilon_train=0.1",' +
            ' "create_atari_environment.game_name="Pong"").')
    
    args = parser.parse_args()
    load_gin_configs(args.config_file, args.gin_bindings)
    
    def option_agent_phi(x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        x = (x/255.0).float()
        return x
    
    if args.mask is True:
        base_dir = os.path.join(args.base_dir, "mask")
    else:
        base_dir = args.base_dir
    
    env = CorridorWrapper(environment_builder('MiniGrid-KeyCorridorS3R3',
                                              seed=args.seed))
    
    
    experiment = DivDisMetaMaskedPPOExperiment(base_dir=base_dir,
                                               seed=args.seed,
                                               option_policy_phi=option_agent_phi,
                                               agent_phi=option_agent_phi,
                                               action_policy=create_mask_cnn_policy(n_channels=3,
                                                                                    action_space=7),
                                               action_vf=create_cnn_vf(n_channels=3),
                                               option_type="divdis",
                                               option_head_num=1,
                                               num_options=0,
                                               num_primitive_actions=7,
                                               available_actions_function=env.available_actions if args.mask else lambda: [torch.tensor(False, dtype=bool)]*7)
    
    experiment.train_meta_agent(env, args.seed, 1e6)
