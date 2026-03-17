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
from portable.option.vf_transfer.models.ensemble_models import DQNEnsemble

@gin.configurable
class SunriseDQNAgent(Agent):
    def __init__(self,
                 use_gpu,
                 buffer_length,
                 learning_rate,
                 batch_size,
                 policy_phi,
                 discount_rate=0.99,
                 ucb_beta=1.0,
                 target_update_interval=10000,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=100000):
        super().__init__()
        
        if use_gpu == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:{}'.format(use_gpu))
        self.buffer_length = buffer_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.policy_phi = policy_phi
        self.discount_rate = discount_rate
        
        self.ucb_beta = ucb_beta
        self.target_update_interval = target_update_interval
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.model = DQNEnsemble()
        self.model.to(self.device)
        self.target_model = deepcopy(self.model)
        self.target_model.eval().to(self.device)
        
        self.train_rewards = deque(maxlen=200)
        self.runs = 0
        self.step_number = 0
        self.option_runs = 0
        
        self.replay_buffer = replay_buffers.PrioritizedReplayBuffer(
            capacity=buffer_length,
        )
        
        self.replay_updater = ReplayUpdater(
            replay_buffer=self.replay_buffer,
            update_func=self.update,
            batchsize=batch_size,
            episodic_update=False,
            episodic_update_len=None,
            n_times_update=1,
            replay_start_size=1e4,
            update_interval=1
        )
        
        self.policy_optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def act(self, obs):
        with torch.no_grad():
            if type(obs) == np.ndarray:
                obs = torch.from_numpy(obs)
            obs = self.policy_phi(obs).unsqueeze(0).to(self.device)
            
            q_mean, q_std = self.model(obs)
            
            if self.training:
                # Epsilon-greedy with UCB
                epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                          np.exp(-self.step_number / self.epsilon_decay)
                
                if np.random.rand() < epsilon:
                    action = np.random.randint(self.model.num_actions)
                else:
                    ucb_values = q_mean + self.ucb_beta * q_std
                    action = torch.argmax(ucb_values, dim=1).item()
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
        }, os.path.join(dirname, 'sunrise_agent.pt'))
    
    def load(self, dirname):
        checkpoint = torch.load(os.path.join(dirname, 'sunrise_agent.pt'), 
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
            
            # Compute ensemble disagreement for SUNRISE weighting
            # Stack to (num_heads, batch_size, num_actions)
            next_q_stack = torch.stack(batch_pred_next_q, dim=0)
            # Get std across heads for next state
            ensemble_std = next_q_stack.std(dim=0)  # (batch_size, num_actions)
            # Use max action's std as uncertainty measure
            max_actions = next_q_stack.mean(dim=0).argmax(dim=1)
            uncertainty = ensemble_std.gather(1, max_actions.unsqueeze(1))  # (batch_size, 1)
        
        total_loss = []
        all_td_errors = []
        self.policy_optimizer.zero_grad()
        
        for head in range(self.model.num_heads):
            # Compute target Q-value
            next_state_values, _ = torch.max(batch_pred_next_q[head], dim=1, keepdim=True)
            target_q = batch_rewards.unsqueeze(1) + \
                      (1 - batch_dones.unsqueeze(1).float()) * self.discount_rate * next_state_values
            
            # Get predicted Q-value for taken action
            pred_q = batch_pred_q[head].gather(1, batch_action.unsqueeze(1))
            
            # SUNRISE: Downweight high-uncertainty updates
            # Use inverse of uncertainty as weight (high uncertainty = low weight)
            weights = 1.0 / (uncertainty + 1e-3)  # Add epsilon for numerical stability
            weights = weights / weights.mean()  # Normalize to mean 1.0
            
            # Apply prioritized replay weights if available
            if "weights" in exp_batch:
                weights = weights * exp_batch["weights"].unsqueeze(1)
            
            # Weighted Huber loss
            loss = F.smooth_l1_loss(pred_q, target_q, reduction='none')
            weighted_loss = (loss * weights).mean()
            
            total_loss.append(weighted_loss)
            
            # Track TD errors for prioritized replay
            if errors_out is not None:
                td_error = torch.abs(pred_q - target_q).squeeze(1)
                all_td_errors.append(td_error)
        
        # Average loss across all heads
        if len(total_loss) > 0:
            total_loss = sum(total_loss) / self.model.num_heads
            total_loss.backward()
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

