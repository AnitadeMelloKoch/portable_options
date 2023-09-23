import os 
from copy import deepcopy 
import logging
import torch 
import torch.nn as nn 
import torch.optim as optim 
import numpy as np 

from portable.option.ensemble.custom_attention import *
from portable.option.policy.models import LinearQFunction, compute_q_learning_loss
from portable.option.policy import UpperConfidenceBound

logger = logging.getLogger(__name__)

class AttentionValueEnsemble():
    def __init__(self,
                 use_gpu,
                 embedding: AutoEncoder,
                 num_actions,
                 
                 attention_module_num=8,
                 learning_rate=2.5e-4,
                 discount_rate=0.9,
                 c=100,
                 gru_hidden_size=128,
                 divergence_loss_scale=0.005,
                 
                 summary_writer=None,
                 model_name='policy'):
        
        self.attention_num = attention_module_num
        self.num_actions = num_actions
        self.use_gpu = use_gpu
        self.gamma = discount_rate
        self.div_scale = divergence_loss_scale
        self.summary_writer = summary_writer
        self.model_name = model_name
        
        self.embedding = embedding
        
        self.flatten = nn.Flatten()
        
        self.recurrent_memory = nn.GRU(
            input_size=embedding.feature_size,
            hidden_size=gru_hidden_size,
            batch_first=True
        )
        
        self.attentions = nn.ModuleList(
            [AttentionLayer(gru_hidden_size) for _ in range(self.attention_num)]
        )
        self.q_networks = nn.ModuleList(
            [LinearQFunction(in_features=gru_hidden_size, 
                             n_actions=num_actions) for _ in range(self.attention_num)]
        )
        self.target_q_networks = deepcopy(self.q_networks)
        self.target_q_networks.eval()
        
        self.optimizer = optim.Adam(
            list(self.attentions.parameters()) + list(self.q_networks.parameters()) + list(self.recurrent_memory.parameters()),
            lr=learning_rate
        )
        
        self.upper_confidence_bound = UpperConfidenceBound(num_modules=attention_module_num,
                                                           c=c)
        self.action_leader = np.random.choice(attention_module_num)
        
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
        
        torch.save(self.q_networks.state_dict(), os.path.join(path, 'policy_networks.pt'))
        torch.save(self.attentions.state_dict(), os.path.join(path, 'attentions.pt'))
        torch.save(self.recurrent_memory.state_dict(), os.path.join(path, 'gru.pt'))
        self.upper_confidence_bound.save(os.path.join(path, 'upper_conf_bound'))
    
    def load(self, path):
        self.q_networks.load_state_dict(torch.load(os.path.join(path, 'policy_networks.pt')))
        self.attentions.load_state_dict(torch.load(os.path.join(path, 'attentions.pt')))
        self.recurrent_memory.load_state_dict(torch.load(os.path.join(path, 'gru.pt')))
        self.upper_confidence_bound.load(os.path.join(path, 'upper_conf_bound'))
    
    def step(self):
        self.upper_confidence_bound.step()
    
    def update_accumulated_rewards(self, reward):
        self.upper_confidence_bound.update_accumulated_rewards(self.action_leader,
                                                               reward)
    
    def update_leader(self):
        self.action_leader = self.upper_confidence_bound.select_leader()
    
    def get_attention_masks(self):
        masks = []
        for attention in self.attentions:
            masks.append(attention.mask())
        
        return masks
    
    def apply_attentions(self, x):
        batch_size, num_features, gru_out_size = x.shape
        attentioned_embeddings = torch.zeros((batch_size, self.attention_num, gru_out_size))
        
        for idx, attention in enumerate(self.attentions):
            out = attention(x)
            attentioned_embeddings[:,idx,...] = out.squeeze(1)
        
        if self.use_gpu:
            return attentioned_embeddings.to("cuda")
        else:
            return attentioned_embeddings
    
    def train(self,
              exp_batch,
              errors_out=None,
              update_target_network=False):
        """
        update both the embedding network and the value network by backproping
        the sumed divergence and q learning loss
        """
        self.q_networks.train()
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
        att_state_embeddings = self.apply_attentions(state_embeddings)
        
        masks = self.get_attention_masks()
        
        # divergence loss
        l_div = 0
        for idx in range(self.attention_num):
            l_div += divergence_loss(masks, idx)
        l_div *= self.div_scale
        loss += l_div
        
        # q learning loss
        td_losses = np.zeros((self.attention_num,))
        with torch.no_grad():
            next_state_embeddings = self.embedding.feature_extractor(batch_next_states)
            next_state_embeddings = next_state_embeddings.unsqueeze(1)
            next_state_embeddings, _ = self.recurrent_memory(next_state_embeddings)
            attn_next_state_embeddings = self.apply_attentions(next_state_embeddings)
        
        all_errors_out = np.zeros((self.attention_num, len(batch_states)))
        
        for idx in range(self.attention_num):
            # predicted q values
            state_attention = att_state_embeddings[:,idx,:]
            batch_pred_q_all_actions = self.q_networks[idx](state_attention)
            batch_pred_q = batch_pred_q_all_actions.evaluate_actions(batch_actions)

            # target q values
            with torch.no_grad():
                next_state_attention = attn_next_state_embeddings[:,idx,:]
                batch_next_state_q_all_actions = self.target_q_networks[idx](next_state_attention)
                next_state_values = batch_next_state_q_all_actions.max
                batch_q_target = batch_rewards + self.gamma*(1-batch_dones)*next_state_values
            
            # loss
            td_loss = compute_q_learning_loss(exp_batch,
                                              batch_pred_q,
                                              batch_q_target,
                                              errors_out=errors_out)
            all_errors_out[idx] = errors_out
            loss += td_loss 
            td_losses[idx] = td_loss.item()
        
        #update errors_out to account for all modules
        del errors_out[:]
        avg_errors_out = np.mean(all_errors_out, axis=0)
        for e in avg_errors_out:
            errors_out.append(e)
        
        # update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # update target network
        if update_target_network:
            self.target_q_networks.load_state_dict(self.q_networks.state_dict())
        
        # logging
        # print(f"Div loss: {l_div.item()}. Q loss: {np.sum(td_losses)}")
        # logger.info(f"Div loss: {l_div.item()}. Q loss: {np.sum(td_losses)}")
        
        if self.summary_writer is not None:
            self.summary_writer.add_scalar('{}/div_loss'.format(self.model_name),
                                           l_div.item(),
                                           self.timestep)
            self.summary_writer.add_scalar('{}/td_loss_total'.format(self.model_name),
                                           np.sum(td_losses),
                                           self.timestep)
            for idx in range(self.attention_num):
                self.summary_writer.add_scalar('{}/td_loss/{}'.format(self.model_name, idx),
                                               td_losses[idx],
                                               self.timestep)
        self.timestep += 1
    
    def predict_actions(self, state, return_q_values=False):
        """
        given a state, each one in the ensemble predicts an action
        args:
            return_q_values: if True, return the predicted q values each learner predicts on the action of their choice.
        """
        with torch.no_grad():
            embeddings = self.embedding.feature_extractor(state)
            self.recurrent_memory.flatten_parameters()
            embeddings = embeddings.unsqueeze(1)
            
            embeddings, _ = self.recurrent_memory(embeddings)
            embeddings = self.apply_attentions(embeddings)
            
            actions = np.zeros(self.attention_num, dtype=np.int32)
            q_values = np.zeros(self.attention_num, dtype=np.float32)
            all_q_values = np.zeros((self.attention_num, self.num_actions),
                                    dtype=np.float32)
            for idx in range(self.attention_num):
                attention = embeddings[:, idx, :]
                q_vals = self.q_networks[idx](attention)
                actions[idx] = q_vals.greedy_actions
                q_values[idx] = q_vals.max
                all_q_values[idx, :] = q_vals.q_values.cpu().numpy()
            
            selected_action = actions[self.action_leader]
        
        if return_q_values:
            return selected_action, actions, q_values, all_q_values
        return selected_action
    
    
    