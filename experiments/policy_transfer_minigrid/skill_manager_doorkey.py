"""Skill manager experiment for MiniGrid DoorKey environment.

Trains a SkillManager sequentially across multiple task seeds. On the first episode
of each task the manager routes to the most confident cluster (if any exist), then
trains with the λ-weighted bootstrap target. After training, collected transitions
are used to assign the task to a cluster via KL divergence and update that cluster's
MetaVF.
"""
import argparse
import sys
import os
import torch
import numpy as np
from portable.utils.utils import load_gin_configs
from experiments.minigrid.utils import environment_builder
from experiments.policy_transfer_minigrid.core.ensemble_dqn_minigrid_experiment import EnsembleDQNMinigridExperiment
from portable.option.vf_transfer.models.skill_manager import SkillManager


def policy_phi(x):
    if type(x) == np.ndarray:
        x = torch.from_numpy(x)
    x = (x / 255.0).float()
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--task_seeds", nargs='+', type=int, required=True,
                        help="List of env seeds to train on sequentially")
    parser.add_argument("--config_file", nargs='+', type=str, required=True)
    parser.add_argument("--gin_bindings", default=[],
                        help='Gin bindings to override config values.')

    args = parser.parse_args()

    load_gin_configs(args.config_file, args.gin_bindings)

    experiment = EnsembleDQNMinigridExperiment(
        base_dir=args.base_dir,
        experiment_name="skill_manager_doorkey",
        seed=args.seed,
        policy_phi=policy_phi,
        use_gpu=0,
        make_videos=False
    )

    log_dir = os.path.join(args.base_dir, "skill_manager_doorkey", str(args.seed), "logs")
    with open(os.path.join(log_dir, "command.txt"), "w") as f:
        f.write(" ".join(sys.argv) + "\n")

    skill_manager = SkillManager(use_gpu=0, policy_phi=policy_phi)

    tasks = [
        (
            environment_builder(
                'MiniGrid-DoorKey-8x8-v0',
                seed=task_seed,
                grayscale=True,
                normalize_obs=False,
                max_steps=2000,
                scale_obs=True,
                final_image_size=(84, 84),
            ),
            task_seed,
        )
        for task_seed in args.task_seeds
    ]

    all_successes = experiment.train_skill_manager(
        skill_manager=skill_manager,
        tasks=tasks,
        steps_per_task=3e6,
        collection_episodes=50,
        eval_interval=10000,
        eval_episodes=20,
    )

    experiment.save_results(all_successes, filename="skill_manager_success_rates.npy")
