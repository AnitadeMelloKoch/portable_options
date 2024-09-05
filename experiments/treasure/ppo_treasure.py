import gym
import gym_treasure_game
from experiments.core.divdis_meta_masked_ppo_experiment import DivDisMetaMaskedPPOExperiment
import argparse
from portable.utils.utils import load_gin_configs
import torch
import os
from portable.agent.model.maskable_ppo import create_mask_linear_atari_model
from experiments.treasure.treasure_wrapper import TreasureInfoWrapper

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
        return torch.tensor(x)
    
    if args.mask is True:
        base_dir = os.path.join(args.base_dir, "mask")
    else:
        base_dir = args.base_dir
    
    env = gym.make('treasure_game-v0')
    env = TreasureInfoWrapper(env)
    def available_actions():
        mask = env.unwrapped.available_mask
        list_mask = [torch.tensor(x == 1, dtype=bool) for x in mask]
        return list_mask
    
    experiment = DivDisMetaMaskedPPOExperiment(base_dir=base_dir,
                                               seed=args.seed,
                                               option_policy_phi=option_agent_phi,
                                               agent_phi=option_agent_phi,
                                               action_model=create_mask_linear_atari_model(9, 9),
                                               option_type="divdis",
                                               option_head_num=1,
                                               num_options=0,
                                               num_primitive_actions=9,
                                               available_actions_function=available_actions)
    
    experiment.train_meta_agent(env, args.seed, 1e6)
