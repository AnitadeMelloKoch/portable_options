import logging 
import os 
import numpy as np 
import gin
import random 
import torch 
import pickle 

from collections import deque

from portable.option.vf_transfer.policy.ensemble_dqn import EnsembleDQNAgent
from portable.option.policy.agents import evaluating
from portable.option.policy.intrinsic_motivation.tabular_count import TabularCount
import matplotlib.pyplot as plt
from portable.option.vf_transfer.models.meta_value_function import MetaVF

from torch.utils.tensorboard import SummaryWriter

@gin.configurable
class VFTransferAgent():
    def __init__(self,
                 use_gpu,
                 log_dir,
                 save_dir,
                 policy_phi,
                 plot_dir,
                 uncertainty_threshold=1.0):
        
        self.gpu = use_gpu
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.policy_phi = policy_phi
        self.plot_dir = plot_dir
        
        self.writer = SummaryWriter(log_dir=log_dir)
        self.uncertainty_threshold = uncertainty_threshold

        self.seed = None
        self.policy = None

        self.meta_vf = MetaVF()
        self.meta_vf.policy_phi = policy_phi
        self._meta_vf_ready = False

        self.reset_policy()

        self.steps = 0

    def reset_policy(self, epsilon_start=None, epsilon_decay=None):
        self.policy = EnsembleDQNAgent(
            use_gpu=self.gpu,
            policy_phi=self.policy_phi
        )
        if epsilon_start is not None:
            self.policy.epsilon_start = epsilon_start
        if epsilon_decay is not None:
            self.policy.epsilon_decay = epsilon_decay
        self.policy.set_meta_vf(self.meta_vf)
        self.policy.meta_vf_ready = self._meta_vf_ready
        self.policy.writer = self.writer
        self.steps = 0
    
    def save(self, save_dir=None):
        if save_dir is None:
            save_dir = self.save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.policy.save(os.path.join(save_dir, 'policy'))
        torch.save({
            'model_state_dict': self.meta_vf.value_function.state_dict(),
            'optimizer_state_dict': self.meta_vf.optimizer.state_dict(),
        }, os.path.join(save_dir, 'meta_vf.pt'))

    def load(self, save_dir=None):
        if save_dir is None:
            save_dir = self.save_dir
        self.policy.load(os.path.join(save_dir, 'policy'))
        checkpoint = torch.load(os.path.join(save_dir, 'meta_vf.pt'),
                                map_location=self.meta_vf.device)
        self.meta_vf.value_function.load_state_dict(checkpoint['model_state_dict'])
        self.meta_vf.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def run_episode(self, env, return_states=False):
        steps = 0
        extrinsic_rewards = []
        states = []
        infos = []
        done = False
        
        state, info = env.reset()

        while not done:
            action = self.policy.act(state)
            next_state, reward, done, info = env.step(action)
            extrinsic_rewards.append(reward)
            states.append(next_state)
            infos.append(info)

            steps += 1
            self.steps += 1
            self.policy.observe(state,
                                action,
                                reward,
                                next_state,
                                done)

            state = next_state

        if self.writer is not None:
            self.writer.add_scalar('episode_length', steps, self.steps)
            self.writer.add_scalar('episode_rewards', sum(extrinsic_rewards), self.steps)
            self.writer.add_scalar('epsilon', self.policy._compute_epsilon(), self.steps)
            if self.policy.task_error_buffer:
                self.writer.add_scalar(
                    'bellman_error', np.mean(self.policy.task_error_buffer), self.steps
                )

        if return_states:
            return extrinsic_rewards, steps, states, infos
        else:
            return extrinsic_rewards, steps
    
    def run_data_collection(self, env, num_episodes, task_id, collection_epsilon=0.3):
        """Collect transitions for meta VF training.

        Returns:
            plot_buffer: list of (obs, x, y, door_open), one per visited cell
            transitions: dict with keys states, next_states, rewards, terminals,
                         uncertainties, task_id — ready for update_meta_vf
        """
        self.policy.training = False
        cell_to_obs = {}
        all_states, all_next_states = [], []
        all_rewards, all_terminals, all_uncertainties = [], [], []
        all_v_means = []
        seen = set()

        for ep_idx in range(num_episodes):
            episode_epsilon = collection_epsilon * (1 - ep_idx / num_episodes)

            episode_states, episode_next_states = [], []
            episode_rewards, episode_terminals, episode_uncertainties = [], [], []
            episode_v_means = []

            done = False
            state, info = env.reset()

            while not done:
                x, y = info['player_pos']
                has_key = info['key_collected']
                cell_to_obs[(x, y, has_key)] = state

                if np.random.rand() < episode_epsilon:
                    action = np.random.randint(self.policy.model.num_actions)
                else:
                    action = self.policy.act(state)

                next_state, reward, done, info = env.step(action)
                v_mean, uncertainty = self.policy.value_function(state)

                transition_key = (state.tobytes(), action, next_state.tobytes())
                if uncertainty < self.uncertainty_threshold and transition_key not in seen:
                    seen.add(transition_key)
                    episode_states.append(state)
                    episode_next_states.append(next_state)
                    episode_rewards.append(reward)
                    episode_terminals.append(done)
                    episode_uncertainties.append(uncertainty)
                    episode_v_means.append(v_mean)

                state = next_state

            all_states.extend(episode_states)
            all_next_states.extend(episode_next_states)
            all_rewards.extend(episode_rewards)
            all_terminals.extend(episode_terminals)
            all_uncertainties.extend(episode_uncertainties)
            all_v_means.extend(episode_v_means)

        self.policy.training = True
        plot_buffer = [(obs, x, y, d) for (x, y, d), obs in cell_to_obs.items()]
        transitions = dict(
            states=all_states,
            next_states=all_next_states,
            rewards=all_rewards,
            terminals=all_terminals,
            uncertainties=all_uncertainties,
            v_means=all_v_means,
            task_id=task_id,
        )
        return plot_buffer, transitions

    def update_meta_vf(self, transitions):
        """Add transitions to the meta VF buffer and train on all accumulated data.

        On the first task (empty buffer) distills V_meta to match V_task scale before
        running TD updates. Subsequent tasks skip distillation and rely on the
        accumulated buffer to maintain the scale.
        """
        is_first_task = len(self.meta_vf.buffer) == 0
        self._meta_vf_ready = True

        self.meta_vf.add_experience(
            transitions['states'],
            transitions['next_states'],
            transitions['rewards'],
            transitions['terminals'],
            transitions['uncertainties'],
            task_id=transitions['task_id'],
        )

        if is_first_task:
            self.meta_vf.distill(
                transitions['states'],
                transitions['v_means'],
                transitions['uncertainties'],
            )

        self.meta_vf.train()

    def initiation(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        obs = self.policy_phi(state).unsqueeze(0)
        with torch.no_grad():
            _, uncertainty = self.meta_vf.query(obs)
        prob = torch.exp(-uncertainty).mean().item()
        return prob

@gin.configurable
class OracleVFAgent():
    """Train a single policy on all tasks simultaneously as an oracle V_meta.

    After training, exposes two queries that can both be plotted side-by-side:
      query_oracle   — the DQN's own value function (true upper bound)
      query_meta_vf  — a MetaVF distilled from the oracle (tests MetaVF capacity)

    Call distill_meta_vf(envs) after train_policy to populate the MetaVF.
    """

    def __init__(self,
                 use_gpu,
                 log_dir,
                 save_dir,
                 policy_phi,
                 plot_dir):
        self.gpu = use_gpu
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.policy_phi = policy_phi
        self.plot_dir = plot_dir

        self.writer = SummaryWriter(log_dir=log_dir)

        self.oracle_policy = EnsembleDQNAgent(
            use_gpu=use_gpu,
            policy_phi=policy_phi,
            use_meta_vf=False,
            epsilon_decay=4e6
        )
        self.oracle_policy.set_meta_vf(None)
        self.steps = 0
        
        self.policy = EnsembleDQNAgent(
            use_gpu=use_gpu,
            policy_phi=policy_phi
        )

        self.meta_vf = MetaVF()
        self.meta_vf.policy_phi = policy_phi

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        self.oracle_policy.save(os.path.join(self.save_dir, 'oracle_policy'))
        self.meta_vf.save(self.save_dir)

    def load(self):
        self.oracle_policy.load(os.path.join(self.save_dir, 'oracle_policy'))

    def run_oracle_policy(self, env):
        steps = 0
        extrinsic_rewards = []
        done = False
        state, info = env.reset()
        while not done:
            action = self.oracle_policy.act(state)
            next_state, reward, done, info = env.step(action)
            extrinsic_rewards.append(reward)
            steps += 1
            self.steps += 1
            self.oracle_policy.observe(state, action, reward, next_state, done)
            state = next_state 
        return extrinsic_rewards, steps

    def run_episode(self, env):
        steps = 0
        extrinsic_rewards = []
        done = False
        state, info = env.reset()
        while not done:
            action = self.policy.act(state)
            next_state, reward, done, info = env.step(action)
            extrinsic_rewards.append(reward)
            steps += 1
            self.steps += 1
            self.policy.observe(state, action, reward, next_state, done)
            state = next_state
        if self.writer is not None:
            self.writer.add_scalar('episode_length', steps, self.steps)
            self.writer.add_scalar('episode_rewards', sum(extrinsic_rewards), self.steps)
        return extrinsic_rewards, steps

    def distill_meta_vf(self, envs, n_episodes=50, collection_epsilon=0.3,
                        uncertainty_threshold=1.0):
        """Collect oracle V estimates from all envs and distill into MetaVF.

        Runs data collection across every env using the trained oracle policy,
        then runs distillation + TD so that MetaVF mirrors the oracle.
        """
        self.oracle_policy.training = False
        all_states, all_means, all_uncertainties = [], [], []

        for env in envs:
            for ep_idx in range(n_episodes):
                episode_epsilon = collection_epsilon * (1 - ep_idx / n_episodes)
                episode_states, episode_next_states = [], []
                episode_rewards, episode_terminals, episode_uncertainties = [], [], []
                seen = set()
                done = False
                state, info = env.reset()
                while not done:
                    if np.random.rand() < episode_epsilon:
                        action = np.random.randint(self.oracle_policy.model.num_actions)
                    else:
                        action = self.oracle_policy.act(state)
                    next_state, reward, done, info = env.step(action)
                    v_mean, uncertainty = self.oracle_policy.value_function(state)
                    key = (state.tobytes(), action, next_state.tobytes())
                    if uncertainty < uncertainty_threshold and key not in seen:
                        seen.add(key)
                        episode_states.append(state)
                        episode_next_states.append(next_state)
                        episode_rewards.append(reward)
                        episode_terminals.append(done)
                        episode_uncertainties.append(uncertainty)
                        all_states.append(state)
                        all_means.append(v_mean)
                        all_uncertainties.append(uncertainty)
                    state = next_state
                if episode_states:
                    self.meta_vf.add_experience(
                        episode_states, episode_next_states,
                        episode_rewards, episode_terminals,
                        episode_uncertainties, task_id=id(env),
                    )

        self.oracle_policy.training = True
        self.meta_vf.distill(all_states, all_means, all_uncertainties)
        self.meta_vf.train()

    def reset_policy(self, meta_vf=None):
        """Create a fresh per-task policy and optionally inject a meta_vf."""
        self.policy = EnsembleDQNAgent(
            use_gpu=self.gpu,
            policy_phi=self.policy_phi,
        )
        if meta_vf is not None:
            self.policy.set_meta_vf(meta_vf)
        self.steps = 0

    def query_oracle(self, obs_tensor):
        """Oracle DQN value — no MetaVF involved."""
        with torch.no_grad():
            q_mean, q_std = self.oracle_policy.model(obs_tensor.to(self.oracle_policy.device))
            best = q_mean.argmax(dim=1, keepdim=True)
            return q_mean.gather(1, best), q_std.gather(1, best)

    def query_meta_vf(self, obs_tensor):
        """MetaVF distilled from the oracle."""
        return self.meta_vf.query(obs_tensor)
