import argparse
import sys
import os
import logging
import torch
import numpy as np
from portable.utils.utils import load_gin_configs
from experiments.minigrid.utils import environment_builder
from experiments.policy_transfer_minigrid.core.ensemble_dqn_minigrid_experiment import EnsembleDQNMinigridExperiment
from portable.option.vf_transfer.vf_transfer_option import OracleVFAgent

# Toggle: True  → distill oracle values into MetaVF, use MetaVF for transfer
#         False → inject oracle policy directly as meta_vf for transfer
USE_DISTILLED_META_VF = False


def policy_phi(x):
    if type(x) == np.ndarray:
        x = torch.from_numpy(x)
    x = (x / 255.0).float()
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--task_seeds", nargs='+', type=int, required=True)
    parser.add_argument("--config_file", nargs='+', type=str, required=True)
    parser.add_argument("--gin_bindings", default=[], help='Gin bindings to override config values.')
    args = parser.parse_args()

    load_gin_configs(args.config_file, args.gin_bindings)

    experiment = EnsembleDQNMinigridExperiment(
        base_dir=args.base_dir,
        experiment_name="perfect_v_transfer_doorkey",
        seed=args.seed,
        policy_phi=policy_phi,
        use_gpu=0,
        make_videos=False
    )

    log_dir = os.path.join(args.base_dir, "perfect_v_transfer_doorkey", str(args.seed), "logs")
    with open(os.path.join(log_dir, "command.txt"), "w") as f:
        f.write(" ".join(sys.argv) + "\n")

    envs = [
        environment_builder(
            'MiniGrid-DoorKey-8x8-v0',
            seed=task_seed,
            grayscale=True,
            normalize_obs=False,
            max_steps=2000,
            scale_obs=True,
            final_image_size=(84, 84),
        )
        for task_seed in args.task_seeds
    ]

    agent = OracleVFAgent(
        use_gpu=0,
        log_dir=experiment.log_dir,
        save_dir=experiment.save_dir,
        plot_dir=experiment.plot_dir,
        policy_phi=policy_phi,
    )

    oracle_successes = experiment.train_oracle_policy(
        agent,
        envs,
        max_steps=4_000_000,
        eval_interval=10000,
        eval_episodes=20,
    )
    agent.save()
    experiment.save_results(oracle_successes, filename="oracle_success_rates.npy")

    # Collect once after oracle training — large n_episodes to maximise state coverage.
    # _collect_plot_states deduplicates by (x, y, door_open) so extra episodes only
    # add newly-discovered cells.
    state_buffers = {
        task_seed: experiment._collect_plot_states(agent.oracle_policy, env, n_episodes=500)
        for task_seed, env in zip(args.task_seeds, envs)
    }

    if USE_DISTILLED_META_VF:
        # Before distillation — MetaVF still has random weights
        for task_seed, env in zip(args.task_seeds, envs):
            experiment.plot_value_heatmap(
                agent.oracle_policy, env,
                queries={
                    "V_oracle": agent.query_oracle,
                    "V_meta": agent.query_meta_vf,
                },
                tag=f"task_{task_seed}_before_distill",
                state_buffer=state_buffers[task_seed],
            )

        agent.distill_meta_vf(envs, n_episodes=50)
        transfer_meta_vf = agent.meta_vf
        meta_vf_label = "V_meta_oracle"

        # After distillation — same states, trained MetaVF
        for task_seed, env in zip(args.task_seeds, envs):
            experiment.plot_value_heatmap(
                agent.oracle_policy, env,
                queries={
                    "V_oracle": agent.query_oracle,
                    "V_meta_oracle": agent.query_meta_vf,
                },
                tag=f"task_{task_seed}_after_distill",
                state_buffer=state_buffers[task_seed],
            )
    else:
        transfer_meta_vf = agent.oracle_policy
        meta_vf_label = "V_oracle"

    transfer_successes = {}
    for task_seed, env in zip(args.task_seeds, envs):
        logging.info(f"Transfer training on task {task_seed}...")

        agent.reset_policy(meta_vf=transfer_meta_vf)

        successes = experiment.train_policy(
            agent,
            [env],
            max_steps=3_000_000,
            eval_interval=10000,
            eval_episodes=20,
        )
        transfer_successes[task_seed] = successes

        def task_vf(obs_tensor):
            q_mean, q_std = agent.policy.model(obs_tensor)
            best = q_mean.argmax(dim=1, keepdim=True)
            return q_mean.gather(1, best), q_std.gather(1, best)

        experiment.plot_value_heatmap(
            agent.policy, env,
            queries={
                "V_task": task_vf,
                meta_vf_label: transfer_meta_vf.query,
            },
            tag=f"task_{task_seed}",
            state_buffer=state_buffers[task_seed],
        )

    experiment.save_results(transfer_successes, filename="transfer_success_rates.npy")
