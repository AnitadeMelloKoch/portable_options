import argparse
import sys
import os
import torch
import numpy as np
from portable.utils.utils import load_gin_configs
from experiments.minigrid.utils import environment_builder
from experiments.policy_transfer_minigrid.core.ensemble_ppo_minigrid_experiment import EnsemblePPOMinigridExperiment
from portable.option.vf_transfer.ppo_skill_library import PPOSkillLibrary


def policy_phi(x):
    if type(x) == np.ndarray:
        x = torch.from_numpy(x)
    x = (x / 255.0).float()
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config_file", nargs='+', type=str, required=True)
    parser.add_argument("--gin_bindings", default=[], help='Gin bindings to override the values'
            ' set in the config files (e.g. "UncertainPPO.epochs=10")')

    args = parser.parse_args()

    load_gin_configs(args.config_file, args.gin_bindings)

    experiment = EnsemblePPOMinigridExperiment(
        base_dir=args.base_dir,
        experiment_name="v_transfer_ppo_doorkey_switch_alpha",
        seed=args.seed,
        policy_phi=policy_phi,
        gpu_id=0,
        make_videos=False,
    )

    log_dir = os.path.join(args.base_dir, "v_transfer_ppo_doorkey_switch_alpha", str(args.seed), "logs")
    with open(os.path.join(log_dir, "command.txt"), "w") as f:
        f.write(" ".join(sys.argv) + "\n")

    skill_library = PPOSkillLibrary(phi=policy_phi)

    task_seeds = [args.seed, args.seed + 1, args.seed + 2, args.seed + 3, args.seed + 4, args.seed + 5, args.seed + 6]
    tasks = [
        (
            lambda s=s: environment_builder(
                'MiniGrid-DoorKey-8x8-v0',
                seed=s,
                grayscale=True,
                normalize_obs=False,
                max_steps=2000,
                scale_obs=True,
                final_image_size=(84, 84),
            ),
            f"doorkey_seed_{s}",
        )
        for s in task_seeds
    ]

    successes = experiment.train_transfer(
        skill_library=skill_library,
        tasks=tasks,
        steps_per_task=int(5e6),
        eval_interval=10000,
        eval_episodes=20,
    )

    experiment.save_results(successes)
