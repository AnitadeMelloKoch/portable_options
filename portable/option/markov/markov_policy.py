import torch
import torch.nn as nn 
import torch.optim as optim 
import numpy as np
from pfrl import explorers
from pfrl import replay_buffers
from pfrl.replay_buffer import ReplayUpdater, batch_experiences
from pfrl.utils.batch_states import batch_states
from copy import deepcopy
import os

from portable.option.ensemble.custom_attention import AutoEncoder, AttentionLayer

from portable.option.policy.agents import Agent 
from portable.option.policy.models import LinearQFunction, compute_q_learning_loss
from pfrl.collections.prioritized import PrioritizedBuffer
import warnings

class MarkovValueModel():
    def __init__(self,
                 use_gpu,
                 embedding: AutoEncoder,
                 num_actions,
                 initial_policy,
                 policy_idx,
                 learning_rate=2.5e-4,
                 discount_rate=0.9,
                 gru_hidden_size=128,
                 summary_writer=None,
                 model_name='markov_policy'):
        
        self.use_gpu = use_gpu
        self.gamma = discount_rate
        self.summary_writer = summary_writer
        self.model_name = model_name 
        self.learning_rate = learning_rate
        
        self.embedding = embedding
        
        self.flatten = nn.Flatten()
        
        self.recurrent_memory = nn.GRU(
            input_size=embedding.feature_size,
            hidden_size=gru_hidden_size,
            batch_first=True
        )
        
        self.recurrent_memory.load_state_dict(initial_policy.recurrent_memory.state_dict())
        
        self.attention = AttentionLayer(gru_hidden_size)
        self.q_network = LinearQFunction(in_features=gru_hidden_size,
                                         n_actions=num_actions)
        
        self.attention = deepcopy(initial_policy.attentions[policy_idx])
        self.q_network.load_state_dict(initial_policy.q_networks[policy_idx].state_dict())
        
        self.target_q_network = deepcopy(self.q_network)
        self.target_q_network.eval()
        
        self.optimizer = optim.Adam(
            list(self.q_network.parameters()),
            lr=learning_rate
        )
        
        self.timestep = 0
    
    def move_to_gpu(self):
        if self.use_gpu:
            self.recurrent_memory.to("cuda")
            self.attentions.to("cuda")
            self.q_networks.to("cuda")
            self.target_q_networks.to("cuda")

    def move_to_cpu(self):
        self.recurrent_memory.to("cpu")
        self.attentions.to("cpu")
        self.q_networks.to("cpu")
        self.target_q_networks.to("cpu")
    
    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        
        torch.save(self.q_network.state_dict(), os.path.join(path, 'policy_networks.pt'))
        torch.save(self.attention.state_dict(), os.path.join(path, 'attentions.pt'))
        torch.save(self.recurrent_memory.state_dict(), os.path.join(path, 'gru.pt'))
    
    def load(self, path):
        if os.path.exists(os.path.join(path, 'policy_networks.pt')):
            self.q_network.load_state_dict(torch.load(os.path.join(path, 'policy_networks.pt')))
            self.attention.load_state_dict(torch.load(os.path.join(path, 'attentions.pt')))
            self.recurrent_memory.load_state_dict(torch.load(os.path.join(path, 'gru.pt')))
        else:
            print("NO PATH TO LOAD ATTENTION VALUE ENSEMBLE FROM")

    def train(self,
              exp_batch,
              errors_out=None,
              update_target_network=False):
        self.q_network.train()
        self.recurrent_memory.flatten_parameters()
        
        batch_states = exp_batch['state']
        batch_actions = exp_batch['action']
        batch_rewards = exp_batch['reward']
        batch_next_states = exp_batch['next_state']
        batch_dones = exp_batch['is_state_terminal']
        
        loss = 0
        
        state_embeddings = self.embedding.feature_extractor(batch_states)
        state_embeddings = state_embeddings.unsqueeze(1)
        state_embeddings, _ = self.recurrent_memory(state_embeddings)
        att_state_embeddings = self.attention(state_embeddings)
        
        with torch.no_grad():
            next_state_embeddings = self.embedding.feature_extractor(batch_next_states)
            next_state_embeddings = next_state_embeddings.unsqueeze(1)
            next_state_embeddings, _ = self.recurrent_memory(next_state_embeddings)
            attn_next_state_embeddings = self.attention(next_state_embeddings)

        batch_pred_q_all_actions = self.q_network(att_state_embeddings)
        batch_pred_q = batch_pred_q_all_actions.evaluate_actions(batch_actions)
        
        # target q values
        with torch.no_grad():
            batch_next_state_q_all_actions = self.q_network(attn_next_state_embeddings)
            next_state_values = batch_next_state_q_all_actions.max
            batch_q_target = batch_rewards + self.gamma*(1-batch_dones)*next_state_values
        
        td_loss = compute_q_learning_loss(exp_batch,
                                          batch_pred_q,
                                          batch_q_target,
                                          errors_out=errors_out)
        
        # update
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()
        
        # update target network
        if update_target_network:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        if self.summary_writer is not None:
            self.summary_writer.add_scalar('{}/td_loss'.format(self.model_name),
                                            td_loss,
                                            self.timestep)
        
        self.timestep += 0
    
    def predict_actions(self, state, return_q_values=False):
        with torch.no_grad():
            embeddings = self.embedding.feature_extractor(state)
            self.recurrent_memory.flatten_parameters()
            embeddings = embeddings.unsqueeze(1)
            
            embeddings, _ = self.recurrent_memory(embeddings)
            embeddings = self.attention(embeddings)
            
            q_vals = self.q_network(embeddings)
            action = q_vals.greedy_actions
            value = q_vals.max
        
        if return_q_values:
            return action, action, value, q_vals.cpu().numpy()
        return action

class MarkovAgent(Agent):
    """
    This agent is initialized from a SINGLE policy of the EnsembleAgent.
    
    This class does not support batched operations
    """
    def __init__(self,
                 use_gpu,
                 warmup_steps,
                 batch_size,
                 phi,
                 prioritized_replay_anneal_steps,
                 embedding,
                 initial_policy,
                 policy_idx,
                 buffer_length=50000,
                 update_interval=4,
                 q_target_update_interval=40,
                 learning_rate=2.5e-4,
                 final_epsilon=0.01,
                 final_exploration_frames=10**4,
                 discount_rate=0.9,
                 num_actions=18,
                 summary_writer=None) -> None:
        super().__init__()
        
        self.use_gpu = use_gpu
        self.phi = phi,
        self.prioritized_replay_anneal_steps = prioritized_replay_anneal_steps
        self.buffer_length = buffer_length
        self.embedding = embedding
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.q_target_update_interval = q_target_update_interval
        self.learning_rate = learning_rate
        self.final_epsilon = final_epsilon
        self.final_exploration_frames = final_exploration_frames
        self.discount_rate = discount_rate
        self.num_actions = num_actions
        self.step_num = 0
        self.episode_number = 0
        
        self.model = MarkovValueModel(use_gpu=use_gpu,
                                      embedding=embedding,
                                      num_actions=num_actions,
                                      initial_policy=initial_policy,
                                      policy_idx=policy_idx,
                                      learning_rate=learning_rate,
                                      discount_rate=discount_rate)
        
        self.explorer = explorers.LinearDecayEpsilonGreedy(
            0.4,
            final_epsilon,
            final_exploration_frames,
            lambda: np.random.randint(num_actions)
        )
        
        self.replay_buffer = replay_buffers.PrioritizedReplayBuffer(capacity=buffer_length,
                                                                    alpha=0.5,
                                                                    beta0=0.4,
                                                                    betasteps=prioritized_replay_anneal_steps,
                                                                    normalize_by_max="memory")
        
        self.replay_updater = ReplayUpdater(replay_buffer=self.replay_buffer,
                                            update_func=self.update,
                                            batchsize=batch_size,
                                            episodic_update=False,
                                            episodic_update_len=None,
                                            n_times_update=1,
                                            replay_start_size=warmup_steps,
                                            update_interval=update_interval)
        
        self.replay_buffer_loaded = True
    
    def move_to_gpu(self):
        self.model.move_to_gpu()
    
    def move_to_cpu(self):
        self.model.move_to_cpu()
    
    def store_buffer(self, save_file):
        self.replay_buffer.save(save_file)
        self.replay_buffer.memory = PrioritizedBuffer()
        
        self.replay_buffer_loaded = False
    
    def load_buffer(self, save_file):
        self.replay_buffer.load(save_file)
        
        self.replay_buffer_loaded = True
    
    def update_step(self):
        self.step_number += 1

    def train(self, epochs):
        if self.replay_buffer_loaded is False:
            raise Exception("replay buffer is not loaded")
        if len(self.replay_buffer) < self.batch_size*epochs:
            return False
        
        for _ in range(epochs):
            transitions = self.replay_buffer.sample(self.batch_size)
            self.replay_updater.update_func(transitions)

    def observe(self,
                obs,
                action,
                reward,
                next_obs,
                terminal,
                update_policy=True):
        if self.replay_buffer_loaded is False:
            warnings.warn("Replay buffer is not loaded. This may not be intended.")
        
        self.update_step()
        
        if self.training:
            transition = {
                "state": obs,
                "action": action,
                "reward": reward,
                "next_state": next_obs,
                "next_action": None,
                "is_state_terminal": terminal
            }
            self.replay_buffer.append(**transition)
            if terminal:
                self.replay_buffer.stop_current_episode()
            
            if update_policy is True:
                self.replay_updater.update_if_necessary(self.step_num)
        if terminal:
            self.episode_number += 1

    def update(self, experiences, errors_out=None):
        if self.replay_buffer_loaded is False:
            warnings.warn("Replay buffer is not loaded. This may not be intended.")
        
        if self.training:
            has_weight = "weight" in experiences[0][0]
            if self.use_gpu:
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            exp_batch = batch_experiences(experiences,
                                          device=device,
                                          phi=self.phi,
                                          gamma=self.discount_rate,
                                          batch_states=batch_states)
            if has_weight:
                exp_batch["weights"] = torch.tensor(
                    [elem[0]["weight"] for elem in experiences],
                    device=device,
                    dtype=torch.float32
                )
                if errors_out is None:
                    errors_out = []
                
                update_target_net = self.step_num%self.q_target_update_interval == 0
                self.model.train(exp_batch,
                                 errors_out,
                                 update_target_net)
                
                if has_weight:
                    assert isinstance(self.replay_buffer, replay_buffers.PrioritizedReplayBuffer)
                    self.replay_buffer.update_errors(errors_out)
            
    def act(self,
            obs,
            return_ensemble_info=False):
        
        if self.replay_buffer_loaded is False:
            warnings.warn("replay buffer is not loaded. This may not be intended")
        
        if return_ensemble_info is True:
            warnings.warn("Markov policy is not an ensemble. Nonsense ensemble info is returned")
        
        if self.use_gpu:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        obs = batch_states([obs], device, self.phi)
        action, actions, val, q_vals = self.model.predict_actions(state=obs, return_q_values=True)
        
        action_selection_func = lambda a: a
        
        if self.training:
            a = self.explorer.select_action(self.step_num,
                                            greedy_action_func=action_selection_func)
        else:
            a = action_selection_func(action)
        
        if return_ensemble_info:
            return a, actions, val
        return a
    
    def save(self, save_dir):
        if self.replay_buffer_loaded is True:
            warnings.warn("Replay buffer is loaded. This may not be intended. Please save replay buffer separately")
        
        self.model.save(save_dir)

    def load(self, load_path):
        self.model.load(load_path)

        self.replay_buffer_loaded = False