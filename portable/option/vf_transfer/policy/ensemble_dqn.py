import os 
import logging 
import gin 
import torch 
import pickle 
import numpy as np
import torch.optim as optim 
import torch.nn as nn
from pfrl import replay_buffers
from pfrl.replay_buffer import ReplayUpdater
logger = logging.getLogger(__name__)
from collections import deque
from copy import deepcopy
import torch.nn.functional as F

from portable.option.policy.agents import Agent
from portable.option.vf_transfer.models.ensemble_models import DQNEnsemble, DQNEnsembleFull

@gin.configurable
class EnsembleDQNAgent(Agent):
    def __init__(self,
                 use_gpu,
                 buffer_length,
                 learning_rate,
                 batch_size,
                 policy_phi,
                 discount_rate=0.99,
                 target_update_interval=10000,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=100000,
                 replay_start_size=10000,
                 use_meta_vf=True,
                 task_buffer_length=100,
                 k=5.0,
                 z0=0.0):
        super().__init__()
        self.k = k
        self.z0 = z0
        
        if use_gpu == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:{}'.format(use_gpu))
        self.buffer_length = buffer_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.policy_phi = policy_phi
        self.discount_rate = discount_rate
        self.use_meta_vf = use_meta_vf
        self.buffer_length = task_buffer_length

        self.target_update_interval = target_update_interval
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.replay_start_size = replay_start_size
        
        self.model = DQNEnsembleFull()
        self.model.to(self.device)
        self.target_model = deepcopy(self.model)
        self.target_model.eval()
        self.policy_optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.meta_vf = None
        self.meta_vf_ready = False
        self.writer = None

        self.train_rewards = deque(maxlen=200)
        self.runs = 0
        self.step_number = 0
        self.option_runs = 0
        
        self.task_value_buffer = deque(maxlen=self.buffer_length)
        self.task_error_buffer = deque(maxlen=self.buffer_length)

        self.replay_buffer = replay_buffers.ReplayBuffer(
            capacity=buffer_length,
        )

        self.replay_updater = ReplayUpdater(
            replay_buffer=self.replay_buffer,
            update_func=self.update,
            batchsize=batch_size,
            episodic_update=False,
            episodic_update_len=None,
            n_times_update=1,
            replay_start_size=replay_start_size,
            update_interval=4
        )
        
    def update_task_error(self, state, next_state, reward):
        if self.meta_vf is None:
            return
        with torch.no_grad():
            s = self.policy_phi(state.float()).unsqueeze(0)
            ns = self.policy_phi(next_state.float()).unsqueeze(0)
            v_state, _ = self.meta_vf.query(s)
            v_next, _ = self.meta_vf.query(ns)
        error = v_state.item() - (reward + self.discount_rate * v_next.item())
        self.task_error_buffer.append(error ** 2)
        self.task_value_buffer.append(v_state.item())
        
    def _compute_epsilon(self):
        step_epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) *
            self.step_number / self.epsilon_decay,
        )
        return step_epsilon

    def act(self, obs):
        with torch.no_grad():
            if type(obs) == np.ndarray:
                obs = torch.from_numpy(obs)
            obs = self.policy_phi(obs).unsqueeze(0).to(self.device)

            q_mean, q_std = self.model(obs)

            if self.training:
                epsilon = self._compute_epsilon()
                if np.random.rand() < epsilon:
                    action = np.random.randint(self.model.num_actions)
                else:
                    action = torch.argmax(q_mean, dim=1).item()
            else:
                action = torch.argmax(q_mean, dim=1).item()

            return action
    
    def value_function(self, obs):
        with torch.no_grad():
            if type(obs) == np.ndarray:
                obs = torch.from_numpy(obs)
            obs = self.policy_phi(obs).unsqueeze(0).to(self.device)
            
            q_mean, q_std = self.model(obs)
            
            best_action = q_mean.argmax(dim=1).item()
            
            value = q_mean[0, best_action].item()
            uncertainty = q_std[0, best_action].item()
            
            return value, uncertainty
        
    
    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.policy_optimizer.state_dict(),
            'step_number': self.step_number,
        }, os.path.join(dirname, 'ensemble_dqn_agent.pt'))
    
    def load(self, dirname):
        checkpoint = torch.load(os.path.join(dirname, 'ensemble_dqn_agent.pt'), 
                               map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_number = checkpoint['step_number']
    
    def end_skill(self, summed_reward):
        self.train_rewards.append(summed_reward)
        self.option_runs += 1
    
    def _storage_phi(self, x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        return x.int()
    
    def set_meta_vf(self, meta_vf):
        self.meta_vf = meta_vf

    def _update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def observe(self, 
                obs, 
                action,
                reward, 
                next_obs, 
                terminal):
        
        self.update_step()
        
        obs = self._storage_phi(obs)
        next_obs = self._storage_phi(next_obs)
        
        self.update_task_error(next_state=next_obs,
                               state=obs,
                               reward=reward)
        
        if self.training:
            transition = {
                "state": obs,
                "action": action,
                "reward": reward,
                "next_state": next_obs,
                "next_action": None,
                "is_state_terminal": terminal,
            }
            
            self.replay_buffer.append(**transition)
            self.replay_updater.update_if_necessary(self.step_number) 
    
    def update_step(self):
        self.step_number += 1
    
    def update(self, experiences, errors_out=None):
        """Update the model using experiences from replay buffer."""
        if self.training:
            from pfrl.replay_buffer import batch_experiences
            from pfrl.utils.batch_states import batch_states
            
            has_weight = "weight" in experiences[0][0]
            
            exp_batch = batch_experiences(
                experiences,
                device=self.device,
                phi=self.policy_phi,
                gamma=self.discount_rate,
                batch_states=batch_states
            )
            
            # Get weights for prioritized experience replay
            if has_weight:
                exp_batch["weights"] = torch.tensor(
                    [elem[0]["weight"] for elem in experiences],
                    device=self.device,
                    dtype=torch.float32
                )
                if errors_out is None:
                    errors_out = []
            
            update_target_net = self.step_number % self.target_update_interval == 0
            self._train_policy(exp_batch, errors_out, update_target_net)
            
            if has_weight:
                assert isinstance(self.replay_buffer, replay_buffers.PrioritizedReplayBuffer)
                self.replay_buffer.update_errors(errors_out)
    
    
    def _train_policy(self,
                      exp_batch,
                      errors_out=None,
                      update_target_network=False):
        batch_obs = exp_batch['state']
        batch_action = exp_batch['action']
        batch_rewards = exp_batch['reward']
        batch_next_obs = exp_batch['next_state']
        batch_dones = exp_batch['is_state_terminal']

        batch_obs = batch_obs.float()
        batch_next_obs = batch_next_obs.float()
        batch_action = batch_action.long()
        
        # Get Q-values from all heads
        batch_pred_q = self.model.forward_all_heads(batch_obs)  # List of (batch_size, num_actions)

        with torch.no_grad():
            batch_pred_next_q = self.target_model.forward_all_heads(batch_next_obs)

            # Compute next-state value target
            stacked_next_q = torch.stack(batch_pred_next_q, dim=0)  # (num_heads, batch, num_actions)
            q_next_mean = stacked_next_q.mean(dim=0)                 # (batch, num_actions)
            q_next_std = stacked_next_q.std(dim=0)                   # (batch, num_actions)
            best_actions = q_next_mean.argmax(dim=1, keepdim=True)   # (batch, 1)
            v_own = q_next_mean.gather(1, best_actions)              # (batch, 1)

            if self.meta_vf is not None and self.use_meta_vf and self.meta_vf_ready:
                v_meta, meta_unc = self.meta_vf.query(batch_next_obs)
                v_meta = v_meta.to(self.device).clamp(-10.0, 10.0)
                meta_unc = meta_unc.to(self.device)
                own_unc = q_next_std.gather(1, best_actions)
                z = torch.log(meta_unc.clamp_min(1e-6)) - torch.log(own_unc.clamp_min(1e-6))
                alpha = torch.sigmoid(-self.k * (z - self.z0))
                v_next = alpha * v_meta + (1 - alpha) * v_own
            else:
                v_next = v_own

        total_loss = []
        all_td_errors = []
        self.policy_optimizer.zero_grad()

        batch_size = batch_obs.shape[0]
        for head in range(self.model.num_heads):
            # Per-head bootstrap mask: each head trains on a random ~50% subset
            mask = torch.bernoulli(torch.full((batch_size,), 0.5, device=self.device)).bool()
            if mask.sum() == 0:
                continue

            target_q = batch_rewards[mask].unsqueeze(1) + \
                      (1 - batch_dones[mask].unsqueeze(1).float()) * self.discount_rate * v_next[mask]

            pred_q = batch_pred_q[head][mask].gather(1, batch_action[mask].unsqueeze(1))

            if "weights" in exp_batch:
                weights = exp_batch["weights"][mask].unsqueeze(1)
                loss = (F.smooth_l1_loss(pred_q, target_q, reduction='none') * weights).mean()
            else:
                loss = F.smooth_l1_loss(pred_q, target_q, reduction='mean')

            total_loss.append(loss)

            # Track TD errors for prioritized replay (over full batch, using unmasked head)
            if errors_out is not None:
                full_pred_q = batch_pred_q[head].gather(1, batch_action.unsqueeze(1))
                full_target_q = batch_rewards.unsqueeze(1) + \
                               (1 - batch_dones.unsqueeze(1).float()) * self.discount_rate * v_next
                td_error = torch.abs(full_pred_q - full_target_q).squeeze(1)
                all_td_errors.append(td_error)
        
        # Average loss across all heads
        if len(total_loss) > 0:
            total_loss = sum(total_loss) / self.model.num_heads
            if self.writer is not None:
                self.writer.add_scalar('policy_loss', total_loss.item(), self.step_number)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.policy_optimizer.step()
            
            # Update prioritized replay errors
            if errors_out is not None:
                # Average TD errors across heads
                avg_td_errors = torch.stack(all_td_errors, dim=0).mean(dim=0)
                del errors_out[:]
                for e in avg_td_errors.detach().cpu().numpy():
                    errors_out.append(float(e))
        
        # Update target network if needed
        if update_target_network:
            self._update_target_network()

    def query(self, batch_obs):
        """This is here in case the policy is used as the meta vf"""
        with torch.no_grad():
            q_mean, q_std = self.model(batch_obs.to(self.device))
            max_a = q_mean.argmax(dim=1, keepdim=True)
            return q_mean.gather(1, max_a), q_std.gather(1, max_a)