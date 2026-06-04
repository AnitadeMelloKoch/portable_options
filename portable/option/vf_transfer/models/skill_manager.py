import numpy as np
import torch
import gin
from typing import List, Optional
from collections import deque

from portable.option.vf_transfer.models.meta_value_function import MetaVF
from portable.option.vf_transfer.policy.ensemble_dqn import EnsembleDQNAgent


@gin.configurable
class SkillManager:

    def __init__(self,
                 use_gpu,
                 policy_phi,
                 divergence_threshold=0.5,
                 min_confidence_threshold=0.1,
                 task_buffer_length=100):
        if use_gpu == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(f'cuda:{use_gpu}')

        self.divergence_threshold = divergence_threshold
        self.min_confidence_threshold = min_confidence_threshold
        self.policy_phi = policy_phi

        self.meta_vfs: List[MetaVF] = []
        self.active_meta_vf_idx: int = None
        self.meta_vf = None

        self._routing_mode: bool = False
        self._first_episode_of_task: bool = True
        self._confidence_sums: List[float] = []
        self._confidence_count: int = 0

        self.policy = EnsembleDQNAgent(use_gpu=use_gpu, policy_phi=policy_phi)

    # ------------------------------------------------------------------
    # Policy management
    # ------------------------------------------------------------------

    def reset_policy(self, epsilon_start=None):
        self.policy = EnsembleDQNAgent(use_gpu=self.policy.device.index if self.policy.device.type != 'cpu' else -1,
                                       policy_phi=self.policy_phi)
        if epsilon_start is not None:
            self.policy.epsilon_start = epsilon_start
        self._set_active_meta_vf()
        self._first_episode_of_task = True

    def _set_active_meta_vf(self):
        meta_vf = self.meta_vfs[self.active_meta_vf_idx] if self.active_cluster_idx is not None else None
        self.policy.set_meta_vf(meta_vf)
        self.meta_vf = meta_vf

    # ------------------------------------------------------------------
    # Episode running with routing
    # ------------------------------------------------------------------

    def run_episode(self, env):
        if self._first_episode_of_task:
            self._start_routing()

        steps = 0
        extrinsic_rewards = []
        done = False
        state, info = env.reset()

        while not done:
            action = self.policy.act(state)

            if self._routing_mode:
                self._track_confidence(state)

            next_state, reward, done, info = env.step(action)
            extrinsic_rewards.append(reward)

            steps += 1
            self.policy.observe(state, action, reward, next_state, done)
            state = next_state

        if self._first_episode_of_task:
            self._end_routing_episode()
            self._first_episode_of_task = False

        return extrinsic_rewards, steps

    # ------------------------------------------------------------------
    # Routing internals
    # ------------------------------------------------------------------

    def _start_routing(self):
        self._routing_mode = len(self.meta_vfs) > 0
        self._confidence_sums = [0.0] * len(self.meta_vfs)
        self._confidence_count = 0
        self.active_meta_vf_idx = None
        self._set_active_meta_vf()

    def _track_confidence(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        obs = self.policy_phi(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            for k, vf in enumerate(self.meta_vfs):
                _, v_std = vf.query(obs)
                self._confidence_sums[k] += (1.0 / (v_std.pow(2) + 1e-8)).mean().item()
        self._confidence_count += 1

    def _end_routing_episode(self):
        self._routing_mode = False
        if not self.meta_vfs:
            return

        avg_confidences = [c / max(self._confidence_count, 1)
                           for c in self._confidence_sums]
        best_k = int(np.argmax(avg_confidences))

        if avg_confidences[best_k] >= self.min_confidence_threshold:
            self.active_meta_vf_idx = best_k
        else:
            self.active_meta_vf_idx = None

        self._set_active_meta_vf()

    # ------------------------------------------------------------------
    # Task assignment
    # ------------------------------------------------------------------

    def assign_task(self, states, dqn_means, dqn_stds):
        """Assign completed task to best matching cluster or seed a new one.

        Args:
            states: Observations from data collection
            dqn_means: V_task mean per state (from DQN ensemble at best action)
            dqn_stds: V_task std per state
        """
        if not self.meta_vfs or not states:
            self._seed_new_vfs()
            return

        states_t = torch.stack([
            torch.tensor(s, dtype=torch.float32) for s in states
        ]).to(self.device)
        means_t = torch.tensor(dqn_means, dtype=torch.float32).to(self.device)
        stds_t = torch.tensor(dqn_stds, dtype=torch.float32).to(self.device)

        divergences = []
        for vf in self.meta_vfs:
            with torch.no_grad():
                v_meta_mean, v_meta_std = vf.query(states_t)
                v_meta_mean = v_meta_mean.squeeze(1)
                v_meta_std = v_meta_std.squeeze(1)
            kl = self._gaussian_kl(means_t, stds_t, v_meta_mean, v_meta_std)
            divergences.append(kl.mean().item())

        best_k = int(np.argmin(divergences))
        if divergences[best_k] < self.divergence_threshold:
            self.active_meta_vf_idx = best_k
        else:
            self._seed_new_vfs()

    def _gaussian_kl(self, mu_p, sigma_p, mu_q, sigma_q):
        """KL(p || q), p = V_task, q = V_meta_k, per state."""
        sigma_q = sigma_q + 1e-8
        sigma_p = sigma_p + 1e-8
        return (torch.log(sigma_q / sigma_p) +
                (sigma_p.pow(2) + (mu_p - mu_q).pow(2)) / (2 * sigma_q.pow(2)) - 0.5)

    def _seed_new_vfs(self):
        self.meta_vfs.append(MetaVF())
        self.active_meta_vf_idx = len(self.meta_vfs) - 1

    # ------------------------------------------------------------------
    # Buffer and training
    # ------------------------------------------------------------------

    def add_experience(self, states, next_states, rewards, terminals, uncertainties, task_id):
        if self.active_meta_vf_idx is not None:
            self.meta_vfs[self.active_meta_vf_idx].add_experience(
                states, next_states, rewards, terminals, uncertainties, task_id
            )

    def train(self):
        if self.active_meta_vf_idx is not None:
            self.meta_vfs[self.active_meta_vf_idx].train()

    def query_active(self, obs):
        """Query the active cluster for evaluation / plotting."""
        if self.active_meta_vf_idx is not None:
            return self.meta_vfs[self.active_meta_vf_idx].query(obs)
        return torch.zeros(1, 1, device=self.device), torch.ones(1, 1, device=self.device)

    def __len__(self):
        return len(self.meta_vfs)
