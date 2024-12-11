from experiments.core.divdis_meta_experiment import DivDisMetaExperiment
import argparse 
from portable.utils.utils import load_gin_configs
import torch 
from experiments.minigrid.utils import environment_builder
from portable.agent.model.ppo import create_cnn_policy, create_cnn_vf
import numpy as np
from experiments.minigrid.advanced_doorkey.core.policy_train_wrapper import LockedRoomPolicyTrainWrapper
import matplotlib.pyplot as plt
from experiments.divdis_minigrid.core.advanced_minigrid_mock_terminations import *
from portable.agent.model.ppo import create_cnn_policy, create_cnn_vf

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
    
    terminations = [
        [PerfectDoorOpen("green")],
        [PerfectDoorOpen("purple")],
        [PerfectDoorOpen("blue")]
    ]
    
    env = LockedRoomPolicyTrainWrapper(environment_builder('LockedRoom-v0',
                            seed=0,
                            grayscale=False,
                            normalize_obs=False,
                            scale_obs=True,
                            final_image_size=(128,128),
                            max_steps=2000))

    experiment = DivDisMetaExperiment(base_dir=args.base_dir,
                                      seed=args.seed,
                                      agent_phi=policy_phi,
                                      option_policy_phi=policy_phi,
                                      option_type="mock",
                                      action_policy=create_cnn_policy(3, 7),
                                      action_vf=create_cnn_vf(3),
                                      terminations=terminations)
    
    experiment.train_option_policies([[[env]],
                                      [[env]],
                                      [[env]]],
                                     0,
                                     2e6)