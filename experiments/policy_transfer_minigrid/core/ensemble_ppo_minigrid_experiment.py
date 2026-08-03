"""Base experiment class for ensemble PPO MinGrid experiments."""
import logging
import datetime
import os
import gin
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
from portable.utils.utils import set_seed
from experiments.experiment_logger import VideoGenerator


@gin.configurable
class EnsemblePPOMinigridExperiment:

    def __init__(self,
                 base_dir,
                 experiment_name,
                 seed,
                 policy_phi,
                 gpu_id,
                 make_videos=False,
                 early_stop_threshold=0.9,
                 early_stop_window=5,
                 n_parallel_envs=1):
        self.name = experiment_name
        self.seed = seed
        self.gpu_id = gpu_id
        self.policy_phi = policy_phi

        self.base_dir = os.path.join(base_dir, experiment_name, str(seed))
        self.log_dir = os.path.join(self.base_dir, "logs")
        self.save_dir = os.path.join(self.base_dir, "checkpoints")
        self.plot_dir = os.path.join(self.base_dir, "plots")

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

        self.early_stop_threshold = early_stop_threshold
        self.early_stop_window = early_stop_window
        self.n_parallel_envs = n_parallel_envs

        set_seed(seed)

        log_file = os.path.join(self.log_dir,
                               "{}.log".format(datetime.datetime.now()))
        logging.basicConfig(filename=log_file,
                           format='%(asctime)s %(levelname)s: %(message)s',
                           level=logging.INFO)

        if make_videos:
            self.video_generator = VideoGenerator(os.path.join(self.base_dir, "videos"))
        else:
            self.video_generator = None

        self.agent = None

    # ------------------------------------------------------------------
    # Single-task training (PPOEnsembleVFAgent)
    # ------------------------------------------------------------------

    def train_policy(self,
                     agent,
                     envs,
                     max_steps=500000,
                     eval_interval=10000,
                     eval_episodes=20):
        self.agent = agent
        step = 0
        episode = 0
        successes = []
        train_rewards = deque(maxlen=200)

        logging.info("Starting training...")
        logging.info(f"Max steps: {max_steps}, Eval interval: {eval_interval}")

        while step < max_steps:
            env = np.random.choice(envs)
            extrinsic_rewards, _, _, episode_steps = agent.run_episode(env)
            train_rewards.append(sum(extrinsic_rewards))
            step += episode_steps
            episode += 1

            if step >= eval_interval * len(successes) + eval_interval:
                eval_success = self.evaluate_policy(agent, envs[0], eval_episodes)
                successes.append((step, eval_success))
                logging.info(f"Step {step}: Success rate = {eval_success:.2%}, "
                             f"Avg train reward = {np.mean(train_rewards):.2f}")

            if episode % 100 == 0:
                logging.info(f"Episode {episode}, Step {step}, "
                             f"Avg reward: {np.mean(train_rewards):.2f}")

        logging.info("Training complete!")
        return successes

    # ------------------------------------------------------------------
    # Multi-task transfer training (PPOSkillLibrary)
    # ------------------------------------------------------------------

    def train_transfer(self,
                       skill_library,
                       tasks,
                       steps_per_task=500000,
                       eval_interval=10000,
                       eval_episodes=20):
        all_successes = {}
        all_alpha_logs = {}

        for task_idx, (env_or_factory, task_id) in enumerate(tasks):
            logging.info(f"Starting task {task_idx + 1}/{len(tasks)}: {task_id}")

            skill_library.add_skill()
            agent = skill_library.skills[-1]

            step = 0
            episode = 0
            successes = []
            train_rewards = deque(maxlen=200)

            eval_env = env_or_factory() if callable(env_or_factory) else env_or_factory

            def task_vf(obs_tensor):
                with torch.no_grad():
                    _, vs = agent.agent.model(obs_tensor)
                    unc = agent.agent.model.vf_uncertainty(obs_tensor)
                return vs, unc

            queries = {"V_task": task_vf}
            if skill_library.meta_vf_ready:
                _meta_vf = skill_library.meta_vf
                def meta_vf_query(obs_tensor):
                    with torch.no_grad():
                        vs = _meta_vf(obs_tensor)
                        unc = _meta_vf.uncertainty(obs_tensor)
                    return vs, unc
                queries["V_meta"] = meta_vf_query

            pre_state_buffer = self._collect_plot_states(agent, eval_env)
            pre_data = self._evaluate_queries(agent, pre_state_buffer, queries)

            if self.n_parallel_envs > 1:
                from pfrl.envs import MultiprocessVectorEnv
                vec_env = MultiprocessVectorEnv(
                    [env_or_factory] * self.n_parallel_envs
                )
                logging.info(f"Task {task_id} | Using {self.n_parallel_envs} parallel envs")

                raw = vec_env.reset()
                # MultiprocessVectorEnv passes through each env's reset() result;
                # with new gymnasium API each is (obs, info) rather than just obs
                if isinstance(raw[0], tuple):
                    obs = np.stack([r[0] for r in raw])
                else:
                    obs = np.array(raw)
                current_returns = [0.0] * self.n_parallel_envs
                episode_successes = deque(maxlen=self.early_stop_window)
                early_stop_triggered = False
                agent.agent.training = True
                parallel_eval_interval = eval_interval * self.n_parallel_envs
                next_episode_log = 100

                while step < steps_per_task and not early_stop_triggered:
                    actions = agent.agent.batch_act(list(obs))
                    step_out = vec_env.step(actions)
                    # handle 4-tuple (old gym) or 5-tuple (new gymnasium terminated+truncated)
                    if len(step_out) == 5:
                        step_obs, rewards, terminated, truncated, _ = step_out
                        dones = [bool(t or tr) for t, tr in zip(terminated, truncated)]
                    else:
                        step_obs, rewards, dones, _ = step_out
                    # pfrl does NOT auto-reset; manually reset done envs.
                    # reset(mask) returns reset obs (tuple) for done envs,
                    # last step obs (array) for others — unwrap either way.
                    if any(dones):
                        mask = np.array([0 if d else 1 for d in dones])
                        reset_results = vec_env.reset(mask=mask)
                        obs = np.stack([
                            r[0] if isinstance(r, tuple) else r
                            for r in reset_results
                        ])
                    else:
                        obs = np.stack([
                            o[0] if isinstance(o, tuple) else o
                            for o in step_obs
                        ])
                    agent.agent.batch_observe(
                        list(obs), list(rewards), list(dones), list(dones)
                    )

                    for i, (r, d) in enumerate(zip(rewards, dones)):
                        current_returns[i] += r
                        if d:
                            train_rewards.append(current_returns[i])
                            episode_successes.append(1.0 if current_returns[i] > 0 else 0.0)
                            current_returns[i] = 0.0
                            episode += 1
                            if (len(episode_successes) == self.early_stop_window and
                                    np.mean(episode_successes) >= self.early_stop_threshold):
                                logging.info(
                                    f"Task {task_id} | Early stop: "
                                    f"success >= {self.early_stop_threshold:.0%} "
                                    f"over last {self.early_stop_window} episodes"
                                )
                                early_stop_triggered = True
                                break

                    step += self.n_parallel_envs

                    if episode > 0 and step >= parallel_eval_interval * len(successes) + parallel_eval_interval:
                        eval_success = self.evaluate_policy(
                            agent, eval_env, eval_episodes
                        )
                        agent.agent.training = True
                        successes.append((step, eval_success))
                        avg_r = f"{np.mean(train_rewards):.2f}" if train_rewards else "N/A"
                        mean_alpha = agent.agent.last_mean_alpha
                        alpha_str = f"{mean_alpha:.4f}" if mean_alpha is not None else "N/A"
                        own_unc = agent.agent.last_own_unc
                        meta_unc = agent.agent.last_meta_unc
                        unc_str = (f" own={own_unc:.4g} meta={meta_unc:.4g}"
                                   if own_unc is not None else "")
                        logging.info(f"Task {task_id} | Step {step}: "
                                     f"Success = {eval_success:.2%}, "
                                     f"Avg reward = {avg_r}, "
                                     f"Alpha = {alpha_str}{unc_str}, "
                                     f"Episodes = {episode}")

                    if episode >= next_episode_log:
                        avg_r = f"{np.mean(train_rewards):.2f}" if train_rewards else "0.00"
                        logging.info(f"Task {task_id} | Episode {episode}, Step {step}, "
                                     f"Avg reward = {avg_r}")
                        next_episode_log = episode + 100

                vec_env.close()

            else:
                env = eval_env

                while step < steps_per_task:
                    extrinsic_rewards, _, _, episode_steps = agent.run_episode(env)
                    train_rewards.append(sum(extrinsic_rewards))
                    step += episode_steps
                    episode += 1

                    if step >= eval_interval * len(successes) + eval_interval:
                        eval_success = self.evaluate_policy(
                            agent, eval_env, eval_episodes
                        )
                        successes.append((step, eval_success))
                        logging.info(f"Task {task_id} | Step {step}: "
                                     f"Success = {eval_success:.2%}, "
                                     f"Avg reward = {np.mean(train_rewards):.2f}")

                        recent = [s for _, s in successes[-self.early_stop_window:]]
                        if (len(recent) == self.early_stop_window and
                                all(s >= self.early_stop_threshold for s in recent)):
                            logging.info(f"Task {task_id} | Early stop: success >= "
                                         f"{self.early_stop_threshold:.0%} for "
                                         f"{self.early_stop_window} evals")
                            break

                    if episode % 100 == 0:
                        logging.info(f"Task {task_id} | Episode {episode}, Step {step}")

            all_successes[task_id] = successes
            all_alpha_logs[task_id] = {
                'alpha': list(agent.agent.alpha_log),
                'update_interval': agent.agent.update_interval,
            }
            logging.info(f"Task {task_id} complete.")

            post_data = self._evaluate_queries(agent, pre_state_buffer, queries)
            combined = {f"{k} pre": v for k, v in pre_data.items()}
            combined.update({f"{k} post": v for k, v in post_data.items()})
            self.plot_combined_heatmap(eval_env, combined, tag=f"task_{task_id}")
            self.plot_alpha_curves({task_id: all_alpha_logs[task_id]})

            if not skill_library.meta_vf_ready:
                logging.info(f"Task {task_id} | Initializing meta VF from task weights...")
                skill_library.consolidate_v()
            else:
                logging.info(f"Task {task_id} | Meta VF already initialized, skipping consolidation.")

        return all_successes

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_policy(self, agent, env, num_episodes=20):
        agent.agent.training = False
        agent.agent.model.eval()
        successes = 0

        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False

            while not done:
                action = agent.act(obs)
                obs, reward, done, _ = env.step(action)
                if reward > 0:
                    successes += 1
                    break

        agent.agent.model.train()
        agent.agent.training = True
        return successes / num_episodes

    # ------------------------------------------------------------------
    # Heatmap plotting
    # ------------------------------------------------------------------

    def _collect_plot_states(self, agent, env, n_episodes=50):
        cell_to_obs = {}
        agent.agent.training = False

        for _ in range(n_episodes):
            obs, info = env.reset()
            done = False
            while not done:
                x, y = info['player_pos']
                has_key = info['key_collected']
                cell_to_obs[(x, y, has_key)] = obs

                action = env.action_space.sample()
                obs, _, done, info = env.step(action)

        agent.agent.training = True
        return [(obs, x, y, k) for (x, y, k), obs in cell_to_obs.items()]

    def _evaluate_queries(self, agent, state_buffer, queries):
        sums = {name: {False: {}, True: {}} for name in queries}
        counts = {name: {False: {}, True: {}} for name in queries}

        phi = agent.agent.phi
        device = agent.agent.device

        agent.agent.training = False
        with torch.no_grad():
            for obs, x, y, has_key in state_buffer:
                obs_tensor = phi(
                    torch.from_numpy(obs) if isinstance(obs, np.ndarray) else obs
                ).unsqueeze(0).to(device)
                for name, query_fn in queries.items():
                    v, u = query_fn(obs_tensor)
                    key = (x, y)
                    if key not in sums[name][has_key]:
                        sums[name][has_key][key] = [0.0, 0.0]
                        counts[name][has_key][key] = 0
                    sums[name][has_key][key][0] += v.item()
                    sums[name][has_key][key][1] += u.item()
                    counts[name][has_key][key] += 1
        agent.agent.training = True

        results = {}
        for name in queries:
            results[name] = {}
            for has_key in (False, True):
                results[name][has_key] = {
                    key: (s[0] / counts[name][has_key][key],
                          s[1] / counts[name][has_key][key])
                    for key, s in sums[name][has_key].items()
                }
        return results

    def plot_combined_heatmap(self, env, panel_data, tag=""):
        env.reset()
        bg_img = env.render()
        img_h, img_w = bg_img.shape[:2]
        grid_w = env.unwrapped.width
        grid_h = env.unwrapped.height
        tile_w = img_w / grid_w
        tile_h = img_h / grid_h

        panel_inches = 6
        pts_per_img_px = 72 * panel_inches / img_w
        marker_size = (tile_w * pts_per_img_px) ** 2 * 0.5

        names = list(panel_data.keys())
        n = len(names)

        for has_key, key_label in [(False, "no_key"), (True, "has_key")]:
            for data_idx, (data_type, cmap) in enumerate([("value", "viridis"),
                                                          ("uncertainty", "plasma")]):
                all_vals = []
                for name in names:
                    for v_mean, u_mean in panel_data[name][has_key].values():
                        all_vals.append(v_mean if data_idx == 0 else u_mean)
                vmin = min(all_vals) if all_vals else 0.0
                vmax = max(all_vals) if all_vals else 1.0

                fig, axes = plt.subplots(1, n, figsize=(panel_inches * n, panel_inches),
                                         squeeze=False)
                axes = axes[0]

                for ax, name in zip(axes, names):
                    ax.imshow(bg_img, origin='upper')
                    cell_data = panel_data[name][has_key]
                    if cell_data:
                        px = [(x + 0.5) * tile_w for x, _ in cell_data]
                        py = [(y + 0.5) * tile_h for _, y in cell_data]
                        vals = [v if data_idx == 0 else u
                                for v, u in cell_data.values()]
                        sc = ax.scatter(px, py, c=vals, cmap=cmap,
                                        vmin=vmin, vmax=vmax,
                                        s=marker_size, alpha=0.75,
                                        edgecolors='white', linewidths=0.5)
                        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                    for i in range(grid_w + 1):
                        ax.axvline(i * tile_w, color='white', linewidth=0.5, alpha=0.4)
                    for j in range(grid_h + 1):
                        ax.axhline(j * tile_h, color='white', linewidth=0.5, alpha=0.4)
                    ax.set_xlim(0, img_w)
                    ax.set_ylim(img_h, 0)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(name)

                suptitle = f"{data_type} — {key_label}" + (f" ({tag})" if tag else "")
                fig.suptitle(suptitle, fontsize=12)
                fname = f"combined_{data_type}_{key_label}{'_' + tag if tag else ''}.png"
                plt.savefig(os.path.join(self.plot_dir, fname), dpi=150, bbox_inches='tight')
                plt.close(fig)
                logging.info(f"Saved combined heatmap: {fname}")

    def plot_alpha_curves(self, alpha_logs):
        for task_id, data in alpha_logs.items():
            if not data['alpha']:
                continue
            steps = [i * data['update_interval'] for i in range(len(data['alpha']))]
            entries = data['alpha']
            if isinstance(entries[0], tuple):
                means = [e[0] for e in entries]
                mins  = [e[1] for e in entries]
                maxs  = [e[2] for e in entries]
            else:
                means = entries
                mins = maxs = None
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(steps, means, label='mean')
            if mins is not None:
                ax.fill_between(steps, mins, maxs, alpha=0.25, label='min–max')
            ax.set_title(f"Alpha — {task_id}")
            ax.set_ylabel("Alpha")
            ax.set_ylim(0, 1)
            ax.set_xlabel("Approximate Step")
            ax.legend(loc='upper right', fontsize=8)
            fname = f"alpha_{task_id}.png"
            plt.savefig(os.path.join(self.plot_dir, fname), dpi=150, bbox_inches='tight')
            plt.close(fig)
            logging.info(f"Saved alpha curves: {fname}")

    # ------------------------------------------------------------------

    def save_results(self, successes, filename="success_rates.npy"):
        save_path = os.path.join(self.save_dir, filename)
        np.save(save_path, np.array(successes, dtype=object))
        logging.info(f"Saved results to {save_path}")
