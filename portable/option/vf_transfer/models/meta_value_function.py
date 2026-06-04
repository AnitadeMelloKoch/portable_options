import torch.nn as nn
import torch
import gin
from copy import deepcopy
from portable.option.vf_transfer.replay_buffer.uncertainty_replay_buffer import UncertaintyReplayBuffer, Transition
import torch.optim as optim
import torch.nn.functional as F
import pickle
import os

class VFHead(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        return self.head(x)

@gin.configurable
class VFEnsemble(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        
        self.num_heads = num_heads
        
        self.shared = nn.Sequential(
            nn.LazyConv2d(out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.heads = nn.ModuleList([VFHead() for _ in range(self.num_heads)])
    
    def forward(self, x):
        out = self.shared(x)
        values = [h(out) for h in self.heads]
        values = torch.stack(values, dim=0)
        value_mean = values.mean(dim=0)
        value_std = values.std(dim=0)
        return value_mean, value_std
        
        

@gin.configurable
class MetaVF():
    def __init__(self,
                 max_capacity,
                 vf_heads,
                 learning_rate,
                 batch_size,
                 epochs_per_update,
                 gamma,
                 distill_epochs=500,
                 target_update_interval=1000,
                 train_uncertainty_threshold=0.1,
                 extra_epochs=100,
                 use_gpu=-1):
        super().__init__()
        self.max_capacity = max_capacity
        self.vf_heads = vf_heads
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs_per_update = epochs_per_update
        self.gamma = gamma
        self.distill_epochs = distill_epochs
        self.target_update_interval = target_update_interval
        self.train_uncertainty_threshold = train_uncertainty_threshold
        self.extra_epochs = extra_epochs

        if use_gpu == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:{}'.format(use_gpu))

        self.buffer = UncertaintyReplayBuffer(max_capacity=max_capacity)
        self.value_function = VFEnsemble(vf_heads)
        self.value_function.to(self.device)
        self.optimizer = optim.Adam(self.value_function.parameters(), lr=learning_rate)
        self.target_value_function = None  # created lazily after first forward pass
        self._train_steps = 0
    
    def save(self, save_dir):
        with open(os.path.join(save_dir, "buffer.pkl"),"wb") as file:
            pickle.dump(self.buffer, file)
        
        torch.save(self.value_function.state_dict(),
                   os.path.join(save_dir, "vf_weights.pth"))
    
    def load(self, save_dir):
        with open(os.path.join(save_dir, "buffer.pkl"), "rb") as file:
            self.buffer = pickle.load(file)
        
        self.value_function.load_state_dict(
            torch.load(os.path.join(save_dir, "vf_weights.pth"))
        )
    
    def query(self, state):
        if not state.is_floating_point():
            state = state.float()
        return self.value_function(state.to(self.device))
    
    def add_experience(self, 
                       states,
                       next_states,
                       rewards,
                       terminals,
                       uncertainties,
                       task_id):
        transitions = [
            Transition(
                uncertainties[idx],
                state=states[idx],
                next_state=next_states[idx],
                reward=rewards[idx],
                terminal=terminals[idx],
                task_id=task_id,
            )
            for idx in range(len(states))
        ]
        self.buffer.add_transitions(transitions)
    
    
    
    def distill(self, states, v_task_means, uncertainties, distill_epochs=None):
        """Directly regress V_meta onto V_task estimates to anchor the value scale.

        Should be called before train() on the first task so that TD bootstrapping
        starts from a sensible scale rather than near-zero random weights.

        Args:
            states: list of observations
            v_task_means: list of scalar V_task(s) estimates from the DQN ensemble
            uncertainties: list of scalar DQN uncertainties used as sample weights
            distill_epochs: number of regression epochs to run
        """
        if distill_epochs is None:
            distill_epochs = self.distill_epochs

        if not states:
            return

        states_t = torch.stack([torch.tensor(s, dtype=torch.float32) for s in states])
        if hasattr(self, 'policy_phi') and self.policy_phi is not None:
            states_t = self.policy_phi(states_t)
        states_t = states_t.to(self.device)
        targets_t = torch.tensor(v_task_means, dtype=torch.float32).unsqueeze(1).to(self.device)
        weights_t = torch.tensor(uncertainties, dtype=torch.float32).to(self.device)
        weights_t = 1.0 / (weights_t.pow(2) + 1e-8)
        weights_t = (weights_t / weights_t.sum()).unsqueeze(1)

        self.value_function.train()
        for _ in range(distill_epochs):
            for start in range(0, len(states_t), self.batch_size):
                end = start + self.batch_size
                w_batch = weights_t[start:end]
                w_batch = w_batch / w_batch.sum()
                self.optimizer.zero_grad()
                loss = self._per_head_loss(
                    states_t[start:end], targets_t[start:end], w_batch
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_function.parameters(), 10.0)
                self.optimizer.step()

        for _ in range(self.extra_epochs):
            epoch_stds = []
            for start in range(0, len(states_t), self.batch_size):
                end = start + self.batch_size
                w_batch = weights_t[start:end]
                w_batch = w_batch / w_batch.sum()
                self.optimizer.zero_grad()
                _, stds = self.value_function(states_t[start:end])
                epoch_stds.append(stds.mean().item())
                loss = self._per_head_loss(
                    states_t[start:end], targets_t[start:end], w_batch
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_function.parameters(), 10.0)
                self.optimizer.step()
            if epoch_stds and (sum(epoch_stds) / len(epoch_stds)) < self.train_uncertainty_threshold:
                break

        # Sync target network to pick up the distilled weights as the new baseline
        self._sync_target()

    def _per_head_loss(self, states, targets, weights):
        """Compute loss against each ensemble head individually.

        Training on the ensemble mean allows heads to cancel each other out
        (one high, one low → zero loss) while maintaining high uncertainty.
        Computing loss per head forces every head to converge to the target.
        """
        feat = self.value_function.shared(states)
        head_vals = torch.stack(
            [h(feat) for h in self.value_function.heads], dim=0
        )  # (n_heads, batch, 1)
        t_exp = targets.unsqueeze(0).expand_as(head_vals)
        w_exp = weights.unsqueeze(0).expand_as(head_vals)
        return (F.smooth_l1_loss(head_vals, t_exp, reduction='none') * w_exp).mean()

    def _sync_target(self):
        self.target_value_function = deepcopy(self.value_function)
        self.target_value_function.eval()

    def train(self):
        if self.target_value_function is None:
            self._sync_target()

        for epoch in range(self.epochs_per_update + self.extra_epochs):
            epoch_stds = []
            for batch in self.buffer.batches(batch_size=self.batch_size):
                self.optimizer.zero_grad()

                states = torch.stack([torch.tensor(t.state, dtype=torch.float32) for t in batch])
                next_states = torch.stack([torch.tensor(t.next_state, dtype=torch.float32) for t in batch])
                if hasattr(self, 'policy_phi') and self.policy_phi is not None:
                    states = self.policy_phi(states)
                    next_states = self.policy_phi(next_states)
                states = states.to(self.device)
                next_states = next_states.to(self.device)
                rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32).to(self.device)
                terminals = torch.tensor([t.terminal for t in batch], dtype=torch.float32).to(self.device)
                t_uncertainties = torch.tensor([t.uncertainty for t in batch], dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    next_values, _ = self.target_value_function(next_states)
                v_targets = rewards.unsqueeze(1) + self.gamma*(1-terminals.unsqueeze(1))*next_values
                _, stds = self.value_function(states)
                epoch_stds.append(stds.mean().item())

                weights = 1.0 / (t_uncertainties.pow(2) + 1e-8)
                weights = (weights / weights.sum()).unsqueeze(1)

                loss = self._per_head_loss(states, v_targets, weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_function.parameters(), 10.0)
                self.optimizer.step()

            if (epoch >= self.epochs_per_update - 1 and epoch_stds and
                    (sum(epoch_stds) / len(epoch_stds)) < self.train_uncertainty_threshold):
                break

        self._sync_target()

