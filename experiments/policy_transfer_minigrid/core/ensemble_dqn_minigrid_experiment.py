"""Base experiment class for SUNRISE MinGrid experiments."""
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
class EnsembleDQNMinigridExperiment:
    """Base experiment class for training SUNRISE agents on MiniGrid environments."""

    def __init__(self,
                 base_dir,
                 experiment_name,
                 seed,
                 policy_phi,
                 use_gpu,
                 make_videos=False,
                 transfer_epsilon_start=0.3,
                 transfer_epsilon_decay=None,
                 early_stop_threshold=0.9,
                 early_stop_window=5):
        self.name = experiment_name
        self.seed = seed
        self.use_gpu = use_gpu
        self.policy_phi = policy_phi

        # Setup directories
        self.base_dir = os.path.join(base_dir, experiment_name, str(seed))
        self.log_dir = os.path.join(self.base_dir, "logs")
        self.save_dir = os.path.join(self.base_dir, 'checkpoints')
        self.plot_dir = os.path.join(self.base_dir, 'plots')

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

        self.transfer_epsilon_start = transfer_epsilon_start
        self.transfer_epsilon_decay = transfer_epsilon_decay
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_window = early_stop_window

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
    # Single-task training (EnsembleDQNAgent)
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
            extrinsic_rewards, episode_steps = agent.run_episode(env)
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

    def train_oracle_policy(self,
                            agent,
                            envs,
                            max_steps=500000,
                            eval_interval=10000,
                            eval_episodes=20):
        """Train the oracle policy on all envs simultaneously.

        Uses agent.run_oracle_policy and evaluates on agent.oracle_policy so
        that the per-task policy (agent.policy) is left untouched.
        """
        self.agent = agent
        step = 0
        episode = 0
        successes = []
        train_rewards = deque(maxlen=200)

        logging.info("Starting oracle policy training...")
        logging.info(f"Max steps: {max_steps}, Eval interval: {eval_interval}")

        while step < max_steps:
            env = np.random.choice(envs)
            extrinsic_rewards, episode_steps = agent.run_oracle_policy(env)
            train_rewards.append(sum(extrinsic_rewards))
            step += episode_steps
            episode += 1

            if step >= eval_interval * len(successes) + eval_interval:
                agent.oracle_policy.training = False
                n_success = 0
                for _ in range(eval_episodes):
                    obs, _ = envs[0].reset()
                    done = False
                    while not done:
                        action = agent.oracle_policy.act(obs)
                        obs, reward, done, info = envs[0].step(action)
                        if reward > 0:
                            n_success += 1
                            break
                agent.oracle_policy.training = True
                eval_success = n_success / eval_episodes
                successes.append((step, eval_success))
                logging.info(f"Oracle | Step {step}: Success rate = {eval_success:.2%}, "
                             f"Avg train reward = {np.mean(train_rewards):.2f}")

            if episode % 100 == 0:
                logging.info(f"Oracle | Episode {episode}, Step {step}, "
                             f"Avg reward: {np.mean(train_rewards):.2f}")

        logging.info("Oracle training complete!")
        agent.save()
        return successes

    # ------------------------------------------------------------------
    # Multi-task transfer training (VFTransferAgent)
    # ------------------------------------------------------------------

    def train_transfer(self,
                       agent,
                       tasks,
                       steps_per_task=500000,
                       collection_episodes=50,
                       eval_interval=10000,
                       eval_episodes=20):
        """Run the full V-transfer pipeline over a sequence of tasks.

        Args:
            agent: VFTransferAgent instance
            tasks: List of (env, task_id) tuples, trained sequentially
            steps_per_task: Training steps before switching to the next task
            collection_episodes: Episodes of data collection after each task
            eval_interval: Steps between evaluations within a task
            eval_episodes: Episodes per evaluation

        Returns:
            Dict mapping task_id to list of (step, success_rate) tuples
        """
        self.agent = agent
        all_successes = {}

        for task_idx, (env, task_id) in enumerate(tasks):
            logging.info(f"Starting task {task_idx + 1}/{len(tasks)}: {task_id}")
            step = 0
            episode = 0
            successes = []
            train_rewards = deque(maxlen=200)

            state_buffer = self._collect_plot_states(agent.policy, env)
            pre_data = self._evaluate_queries(
                state_buffer, {"V_meta_before": agent.meta_vf.query}, agent.policy
            )
            
            state_buffer_dict = {}          

            while step < steps_per_task:
                extrinsic_rewards, episode_steps, episode_states, episode_infos = agent.run_episode(env, return_states=True)
                train_rewards.append(sum(extrinsic_rewards))
                step += episode_steps
                episode += 1
                
                for state, info in zip(episode_states, episode_infos):
                    key = (
                        info['player_x'],
                        info['player_y'],
                        info['door_open'],
                    )
                    state_buffer_dict[key] = state

                if step >= eval_interval * len(successes) + eval_interval:
                    eval_success = self.evaluate_policy(agent, env, eval_episodes)
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

            logging.info(f"Task {task_id} complete. Collecting data...")
            _, transitions = agent.run_data_collection(env, collection_episodes, task_id)

            logging.info(f"Updating meta VF...")
            agent.update_meta_vf(transitions)

            def task_vf(obs_tensor):
                q_mean, q_std = agent.policy.model(obs_tensor)
                best = q_mean.argmax(dim=1, keepdim=True)
                return q_mean.gather(1, best), q_std.gather(1, best)

            state_buffer = [(obs,x,y,d) for (x,y,d), obs in state_buffer_dict.items()]

            post_data = self._evaluate_queries(
                state_buffer,
                {"V_meta_after": agent.meta_vf.query, "V_task": task_vf},
                agent.policy,
            )
            self.plot_combined_heatmap(
                env, {**pre_data, **post_data}, tag=f"task_{task_id}"
            )

            logging.info(f"Resetting policy for next task...")
            epsilon_start = self.transfer_epsilon_start if task_idx < len(tasks) - 1 else None
            agent.reset_policy(epsilon_start=epsilon_start, epsilon_decay=self.transfer_epsilon_decay)

        return all_successes

    # ------------------------------------------------------------------
    # Skill manager training
    # ------------------------------------------------------------------

    def train_skill_manager(self,
                            skill_manager,
                            tasks,
                            steps_per_task=500000,
                            collection_episodes=50,
                            eval_interval=10000,
                            eval_episodes=20,
                            collection_epsilon=0.3,
                            uncertainty_threshold=1.0):
        """Train a SkillManager sequentially across tasks.

        After each task: collect data via the policy, assign the task to a cluster,
        add experience to that cluster's MetaVF, train it, then reset the policy.

        Args:
            skill_manager: SkillManager instance
            tasks: list of (env, task_id) tuples
            steps_per_task: max training steps per task
            collection_episodes: episodes of data collection after each task
            eval_interval: steps between evaluations
            eval_episodes: episodes per evaluation
            collection_epsilon: starting epsilon for data collection (decays to 0)
            uncertainty_threshold: transitions with higher uncertainty are excluded

        Returns:
            Dict mapping task_id to list of (step, success_rate) tuples
        """
        all_successes = {}

        for task_idx, (env, task_id) in enumerate(tasks):
            logging.info(f"[SkillManager] Starting task {task_idx + 1}/{len(tasks)}: {task_id}")
            step = 0
            episode = 0
            successes = []
            train_rewards = deque(maxlen=200)

            while step < steps_per_task:
                rewards, episode_steps = skill_manager.run_episode(env)
                train_rewards.append(sum(rewards))
                step += episode_steps
                episode += 1

                if step >= eval_interval * len(successes) + eval_interval:
                    eval_success = self.evaluate_policy(skill_manager, env, eval_episodes)
                    successes.append((step, eval_success))
                    logging.info(f"[SkillManager] Task {task_id} | Step {step}: "
                                 f"Success = {eval_success:.2%}, "
                                 f"Avg reward = {np.mean(train_rewards):.2f}, "
                                 f"Clusters = {len(skill_manager)}")

                    recent = [s for _, s in successes[-self.early_stop_window:]]
                    if (len(recent) == self.early_stop_window and
                            all(s >= self.early_stop_threshold for s in recent)):
                        logging.info(f"[SkillManager] Task {task_id} | Early stop")
                        break

                if episode % 100 == 0:
                    logging.info(f"[SkillManager] Task {task_id} | Episode {episode}, Step {step}")

            all_successes[task_id] = successes

            # ------ data collection ------
            logging.info(f"[SkillManager] Task {task_id} | Collecting data...")
            policy = skill_manager.policy
            policy.training = False

            states, next_states, rewards_buf = [], [], []
            terminals, uncertainties, means, stds = [], [], [], []
            plot_buffer = []

            for ep_idx in range(collection_episodes):
                episode_epsilon = collection_epsilon * (1 - ep_idx / collection_episodes)
                done = False
                seen = set()
                state, info = env.reset()

                while not done:
                    x, y = info['player_pos']
                    has_key = info['key_collected']
                    plot_buffer.append((state, x, y, has_key))

                    if np.random.rand() < episode_epsilon:
                        action = np.random.randint(policy.model.num_actions)
                    else:
                        action = policy.act(state)

                    next_state, reward, done, info = env.step(action)
                    v_mean, uncertainty = policy.value_function(state)

                    key = (state.tobytes(), action, next_state.tobytes())
                    if uncertainty < uncertainty_threshold and key not in seen:
                        seen.add(key)
                        states.append(state)
                        next_states.append(next_state)
                        rewards_buf.append(reward)
                        terminals.append(done)
                        uncertainties.append(uncertainty)
                        means.append(v_mean)
                        stds.append(uncertainty)

                    state = next_state

            policy.training = True

            # ------ cluster assignment and meta VF update ------
            logging.info(f"[SkillManager] Task {task_id} | Assigning cluster...")
            skill_manager.assign_task(states, means, stds)
            skill_manager.add_experience(states, next_states, rewards_buf,
                                         terminals, uncertainties, task_id)
            skill_manager.train()
            logging.info(f"[SkillManager] Task {task_id} | Clusters: {len(skill_manager)}, "
                         f"active: {skill_manager.active_cluster_idx}")

            def task_vf(obs_tensor):
                q_mean, q_std = skill_manager.policy.model(obs_tensor)
                best = q_mean.argmax(dim=1, keepdim=True)
                return q_mean.gather(1, best), q_std.gather(1, best)

            self.plot_value_heatmap(
                skill_manager.policy, env,
                queries={"V_meta": skill_manager.query_active, "V_task": task_vf},
                tag=f"task_{task_id}",
                state_buffer=plot_buffer,
            )

            epsilon_start = self.transfer_epsilon_start if task_idx < len(tasks) - 1 else None
            skill_manager.reset_policy(epsilon_start=epsilon_start)

        return all_successes

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_policy(self, agent, env, num_episodes=20, eval_epsilon=0.05):
        """Evaluate success rate of the agent's current policy."""
        policy = agent.policy
        policy.training = False
        successes = 0

        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False

            while not done:
                if np.random.rand() < eval_epsilon:
                    action = np.random.randint(policy.model.num_actions)
                else:
                    action = policy.act(obs)
                obs, reward, done, info = env.step(action)

                if reward > 0:
                    successes += 1
                    break

        policy.training = True
        return successes / num_episodes

    # ------------------------------------------------------------------
    # Heatmap plotting
    # ------------------------------------------------------------------

    def _collect_plot_states(self, policy, env, n_episodes=50):
        """Collect one representative observation per (x, y, has_key) cell.

        Runs n_episodes with fully random actions so all reachable cells are
        visited regardless of what the policy has learned. Each unique cell keeps
        only its most recently seen observation.
        """
        cell_to_obs = {}
        policy.training = False

        for _ in range(n_episodes):
            obs, info = env.reset()
            done = False
            while not done:
                x, y = info['player_pos']
                has_key = info['key_collected']
                cell_to_obs[(x, y, has_key)] = obs

                action = np.random.randint(policy.model.num_actions)
                obs, _, done, info = env.step(action)

        policy.training = True
        return [(obs, x, y, k) for (x, y, k), obs in cell_to_obs.items()]

    def _evaluate_queries(self, state_buffer, queries, policy):
        """Evaluate query functions on state_buffer, averaging per (x, y, has_key) cell.

        Returns:
            dict: {name: {has_key: {(x, y): (mean_value, mean_uncertainty)}}}
        """
        sums = {name: {False: {}, True: {}} for name in queries}
        counts = {name: {False: {}, True: {}} for name in queries}

        policy.training = False
        with torch.no_grad():
            for obs, x, y, has_key in state_buffer:
                obs_tensor = policy.policy_phi(
                    torch.from_numpy(obs) if isinstance(obs, np.ndarray) else obs
                ).unsqueeze(0).to(policy.device)
                for name, query_fn in queries.items():
                    v, u = query_fn(obs_tensor)
                    key = (x, y)
                    if key not in sums[name][has_key]:
                        sums[name][has_key][key] = [0.0, 0.0]
                        counts[name][has_key][key] = 0
                    sums[name][has_key][key][0] += v.item()
                    sums[name][has_key][key][1] += u.item()
                    counts[name][has_key][key] += 1
        policy.training = True

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
        """Plot all queries side by side in one figure per door state.

        Uses a shared colour scale across panels so values are directly comparable.
        Produces two figures per door state: one for value (viridis), one for
        uncertainty (plasma).

        Args:
            env: MiniGrid environment for background rendering and grid dimensions.
            panel_data: output of _evaluate_queries —
                        {name: {door_open: {(x,y): (mean_v, mean_u)}}}
            tag: suffix for saved filenames
        """
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
                # Compute shared colour limits across all panels
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
                        px = [(x + 0.5) * tile_w for x, y in cell_data]
                        py = [(y + 0.5) * tile_h for x, y in cell_data]
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

    def plot_value_heatmap(self, policy, env, queries, tag="", state_buffer=None):
        """Superimpose VF scatter plots on the rendered env image.

        Produces one figure per (query_name, door_open) pair. Each figure shows
        the env background with scatter points at cell centres coloured by mean VF
        value, and thin grid lines aligned to tile boundaries.

        Args:
            policy: EnsembleDQNAgent — used for policy_phi and state collection
            env: MiniGrid environment (render_mode='rgb_array')
            queries: dict of {name: query_fn(obs_tensor) -> (value, _)}
            tag: suffix appended to saved filenames
            state_buffer: optional list of (obs, x, y, door_open). If None,
                          1000 states are collected via _collect_plot_states.
        """
        if state_buffer is None:
            state_buffer = self._collect_plot_states(policy, env)

        # Render background at full resolution from a fresh reset
        env.reset()
        bg_img = env.render()       # (H, W, 3) uint8
        img_h, img_w = bg_img.shape[:2]
        grid_w = env.unwrapped.width
        grid_h = env.unwrapped.height
        tile_w = img_w / grid_w
        tile_h = img_h / grid_h

        # Marker area so dots are roughly half a cell in size
        fig_inches = 8
        pts_per_img_px = 72 * fig_inches / img_w
        marker_size = (tile_w * pts_per_img_px) ** 2 * 0.5

        policy.training = False
        with torch.no_grad():
            for name, query_fn in queries.items():
                value_sums = {False: np.zeros((grid_h, grid_w)),
                              True:  np.zeros((grid_h, grid_w))}
                uncert_sums = {False: np.zeros((grid_h, grid_w)),
                               True:  np.zeros((grid_h, grid_w))}
                visit_counts = {False: np.zeros((grid_h, grid_w)),
                                True:  np.zeros((grid_h, grid_w))}

                for obs, x, y, has_key in state_buffer:
                    obs_tensor = policy.policy_phi(
                        torch.from_numpy(obs) if isinstance(obs, np.ndarray) else obs
                    ).unsqueeze(0).to(policy.device)
                    value, uncert = query_fn(obs_tensor)
                    value_sums[has_key][y, x] += value.item()
                    uncert_sums[has_key][y, x] += uncert.item()
                    visit_counts[has_key][y, x] += 1

                for has_key, key_label in [(False, "no_key"), (True, "has_key")]:
                    counts = visit_counts[has_key]

                    px_list, py_list, v_list, u_list = [], [], [], []
                    for y_g in range(grid_h):
                        for x_g in range(grid_w):
                            if counts[y_g, x_g] > 0:
                                v_list.append(value_sums[has_key][y_g, x_g] / counts[y_g, x_g])
                                u_list.append(uncert_sums[has_key][y_g, x_g] / counts[y_g, x_g])
                                px_list.append((x_g + 0.5) * tile_w)
                                py_list.append((y_g + 0.5) * tile_h)

                    for data, label, cmap in [
                        (v_list, f"{name}_value", 'viridis'),
                        (u_list, f"{name}_uncertainty", 'plasma'),
                    ]:
                        fig, ax = plt.subplots(figsize=(fig_inches, fig_inches))
                        ax.imshow(bg_img, origin='upper')

                        if data:
                            sc = ax.scatter(px_list, py_list, c=data, cmap=cmap,
                                            s=marker_size, alpha=0.75,
                                            edgecolors='white', linewidths=0.5)
                            plt.colorbar(sc, ax=ax, label=label, fraction=0.046, pad=0.04)

                        for i in range(grid_w + 1):
                            ax.axvline(i * tile_w, color='white', linewidth=0.5, alpha=0.4)
                        for j in range(grid_h + 1):
                            ax.axhline(j * tile_h, color='white', linewidth=0.5, alpha=0.4)

                        ax.set_xlim(0, img_w)
                        ax.set_ylim(img_h, 0)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_title(f"{label} — {key_label}" + (f" ({tag})" if tag else ""))

                        fname = f"{label}_{key_label}{'_' + tag if tag else ''}.png"
                        plt.savefig(os.path.join(self.plot_dir, fname), dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        logging.info(f"Saved heatmap: {fname}")
        policy.training = True

    # ------------------------------------------------------------------

    def save_results(self, successes, filename="success_rates.npy"):
        save_path = os.path.join(self.save_dir, filename)
        np.save(save_path, np.array(successes, dtype=object))
        logging.info(f"Saved results to {save_path}")
