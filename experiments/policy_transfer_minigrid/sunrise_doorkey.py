"""SUNRISE DQN experiment for MiniGrid DoorKey environment."""
import argparse
import torch
import numpy as np
from portable.utils.utils import load_gin_configs
from experiments.minigrid.utils import environment_builder
from experiments.policy_transfer_minigrid.core.sunrise_minigrid_experiment import SunriseMinigridExperiment
from portable.option.vf_transfer.policy.sunrise import SunriseDQNAgent
from portable.option.vf_transfer.models.ensemble_models import DQNEnsemble


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
    experiment = SunriseMinigridExperiment(
        base_dir=args.base_dir,
        experiment_name="sunrise_doorkey",
        seed=args.seed,
        policy_phi=policy_phi,
        use_gpu=0,
        make_videos=False
    )

    agent = SunriseDQNAgent(
        use_gpu=0,
        buffer_length=100000,
        learning_rate=1e-4,
        batch_size=32,
        policy_phi=policy_phi,
        discount_rate=0.99,
        ucb_beta=1.0,
        target_update_interval=1000,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=100000
    )

    envs = [
        environment_builder('AdvancedDoorKey-8x8-v0', seed=args.seed, grayscale=False, normalize_obs=False, max_steps=1500),
    ]

    # Train the agent
    successes = experiment.train_policy(
        agent=agent,
        envs=envs,
        max_steps=500000,
        eval_interval=10000,
        eval_episodes=20
    )

    # Save results
    experiment.save_results(successes)
