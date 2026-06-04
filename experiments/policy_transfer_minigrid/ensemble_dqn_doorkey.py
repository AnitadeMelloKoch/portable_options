"""SUNRISE DQN experiment for MiniGrid DoorKey environment."""
import argparse
import sys
import os
import torch
import numpy as np
from portable.utils.utils import load_gin_configs
from experiments.minigrid.utils import environment_builder
from experiments.policy_transfer_minigrid.core.ensemble_dqn_minigrid_experiment import EnsembleDQNMinigridExperiment
from portable.option.vf_transfer.vf_transfer_option import VFTransferAgent


def policy_phi(x):
    if type(x) == np.ndarray:
        x = torch.from_numpy(x)
    x = (x/255.0).float()
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config_file", nargs='+', type=str, required=True)
    parser.add_argument("--gin_bindings", default=[], help='Gin bindings to override the values' +
            ' set in the config files (e.g. "DQNAgent.epsilon_train=0.1",' +
            ' "create_atari_environment.game_name="Pong"").')

    args = parser.parse_args()

    # Load gin configs
    load_gin_configs(args.config_file, args.gin_bindings)

    # Create experiment
    experiment = EnsembleDQNMinigridExperiment(
        base_dir=args.base_dir,
        experiment_name="ensemble_dqn_doorkey",
        seed=args.seed,
        policy_phi=policy_phi,
        use_gpu=0,
        make_videos=False
    )

    # Save the full command used to run this experiment
    log_dir = os.path.join(args.base_dir, "ensemble_dqn_doorkey", str(args.seed), "logs")
    with open(os.path.join(log_dir, "command.txt"), "w") as f:
        f.write(" ".join(sys.argv) + "\n")

    agent = VFTransferAgent(
        use_gpu=0,
        log_dir=experiment.log_dir,
        save_dir=experiment.save_dir,
        plot_dir=experiment.plot_dir,
        policy_phi=policy_phi,
    )

    envs = [
        
        environment_builder('MiniGrid-DoorKey-8x8-v0', 
                            seed=args.seed, 
                            grayscale=True, 
                            normalize_obs=False, 
                            max_steps=2000, 
                            scale_obs=True, 
                            final_image_size=(84,84)),
    ]

    # Train the agent
    successes = experiment.train_policy(
        agent=agent,
        envs=envs,
        max_steps=3e6,
        eval_interval=10000,
        eval_episodes=20
    )

    # Save results
    experiment.save_results(successes)
