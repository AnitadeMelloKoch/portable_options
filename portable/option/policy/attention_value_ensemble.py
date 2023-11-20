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
        
        self.recurrent_memories = nn.ModuleList([
            nn.GRU(
                input_size=embedding.feature_size,
                hidden_size=gru_hidden_size,
                batch_first=True
            ) for _ in range(self.attention_num)
        ])
        
        self.attentions = nn.ModuleList(
            [AttentionLayer(gru_hidden_size) for _ in range(self.attention_num)]
        )
        self.q_networks = nn.ModuleList(
            [LinearQFunction(in_features=gru_hidden_size, 
                             n_actions=num_actions) for _ in range(self.attention_num)]
        )
        self.target_q_networks = deepcopy(self.q_networks)
        self.target_q_networks.eval()
        
        self.optimizers = [
            optim.Adam(
                list(self.attentions[i].parameters()) + list(self.q_networks[i].parameters()) + list(self.recurrent_memories[i].parameters()),
                lr=learning_rate
            ) for i in range(self.attention_num)
        ]
        
        self.upper_confidence_bound = UpperConfidenceBound(num_modules=attention_module_num,
                                                           c=c)
        self.action_leader = np.random.choice(attention_module_num)
        
        self.timestep = [0]*self.attention_num
    
    def move_to_gpu(self):
        if self.use_gpu:
            self.recurrent_memories[self.action_leader].to("cuda")
            self.attentions.to("cuda")
            self.q_networks[self.action_leader].to("cuda")
            self.target_q_networks[self.action_leader].to("cuda")
    
    def move_to_cpu(self):
        self.recurrent_memories[self.action_leader].to("cpu")
        self.attentions.to("cpu")
        self.q_networks[self.action_leader].to("cpu")
        self.target_q_networks[self.action_leader].to("cpu")
    
    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        
        torch.save(self.q_networks.state_dict(), os.path.join(path, 'policy_networks.pt'))
        torch.save(self.attentions.state_dict(), os.path.join(path, 'attentions.pt'))
        torch.save(self.recurrent_memory.state_dict(), os.path.join(path, 'gru.pt'))
        self.upper_confidence_bound.save(os.path.join(path, 'upper_conf_bound'))
    
    def load(self, path):
        if os.path.exists(os.path.join(path, 'policy_networks.pt')):
            self.q_networks.load_state_dict(torch.load(os.path.join(path, 'policy_networks.pt')))
            self.attentions.load_state_dict(torch.load(os.path.join(path, 'attentions.pt')))
            self.recurrent_memory.load_state_dict(torch.load(os.path.join(path, 'gru.pt')))
            self.upper_confidence_bound.load(os.path.join(path, 'upper_conf_bound'))
        else:
            print("NO PATH TO LOAD ATTENTION VALUE ENSEMBLE FROM")
    
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
    
    def apply_attention(self, x):
        attentioned_embedding = self.attentions[self.action_leader](x)

        return attentioned_embedding
    
    def train(self,
              exp_batch,
              errors_out=None,
              update_target_network=False):
        """
        update both the embedding network and the value network by backproping
        the sumed divergence and q learning loss
        """
        self.q_networks.train()
        self.recurrent_memories[self.action_leader].flatten_parameters()
        
        batch_states = exp_batch['state']
        batch_actions = exp_batch['action']
        batch_rewards = exp_batch['reward']
        batch_next_states = exp_batch['next_state']
        batch_dones = exp_batch['is_state_terminal']
        
        loss = 0
        
        state_embeddings = self.embedding.feature_extractor(batch_states)
        state_embeddings = state_embeddings.unsqueeze(1)
        state_embeddings, _ = self.recurrent_memories[self.action_leader](state_embeddings)
        state_embeddings = state_embeddings.squeeze()
        att_state_embeddings = self.apply_attention(state_embeddings)
        
        masks = self.get_attention_masks()
        
        # divergence loss
        l_div = divergence_loss(masks, self.action_leader)
        
        l_div *= self.div_scale
        loss += l_div
        
        # q learning loss
        td_losses = np.zeros((self.attention_num,))
        with torch.no_grad():
            next_state_embeddings = self.embedding.feature_extractor(batch_next_states)
            next_state_embeddings = next_state_embeddings.unsqueeze(1)
            next_state_embeddings, _ = self.recurrent_memories[self.action_leader](next_state_embeddings)
            next_state_embeddings = next_state_embeddings.squeeze()
            attn_next_state_embeddings = self.apply_attention(next_state_embeddings)
        
        # predicted q values
        state_attention = att_state_embeddings
        batch_pred_q_all_actions = self.q_networks[self.action_leader](state_attention)
        
        batch_pred_q = batch_pred_q_all_actions.evaluate_actions(batch_actions)

        # target q values
        with torch.no_grad():
            next_state_attention = attn_next_state_embeddings
            batch_next_state_q_all_actions = self.target_q_networks[self.action_leader](next_state_attention)
            next_state_values = torch.argmax(batch_next_state_q_all_actions.q_values, dim=-1)
            batch_q_target = batch_rewards + self.gamma*(1-batch_dones)*next_state_values
            
        # loss
        td_loss = compute_q_learning_loss(exp_batch,
                                            batch_pred_q,
                                            batch_q_target,
                                            errors_out=errors_out)
        loss += td_loss 
        
        # update
        self.optimizers[self.action_leader].zero_grad()
        loss.backward()
        self.optimizers[self.action_leader].step()
        
        # update target network
        if update_target_network:
            self.target_q_networks.load_state_dict(self.q_networks.state_dict())
        
        # logging
        # print(f"Div loss: {l_div.item()}. Q loss: {np.sum(td_losses)}")
        # logger.info(f"Div loss: {l_div.item()}. Q loss: {np.sum(td_losses)}")
        
        if self.summary_writer is not None:
            self.summary_writer.add_scalar('{}/div_loss/{}'.format(self.model_name, self.action_leader),
                                           l_div.item(),
                                           self.timestep[self.action_leader])
            self.summary_writer.add_scalar('{}/td_loss/{}'.format(self.model_name, self.action_leader),
                                            td_losses[self.action_leader],
                                            self.timestep[self.action_leader])
        self.timestep[self.action_leader] += 1
    
    def predict_actions(self, state, return_q_values=False):
        """
        given a state, action leader predicts an action
        args:
            return_q_values: if True, return the predicted q values each learner predicts on the action of their choice.
        """
        with torch.no_grad():
            embeddings = self.embedding.feature_extractor(state)
            self.recurrent_memories[self.action_leader].flatten_parameters()
            embeddings = embeddings.unsqueeze(1)
            
            embeddings, _ = self.recurrent_memories[self.action_leader](embeddings)
            embeddings = embeddings.squeeze()
            embeddings = self.apply_attention(embeddings)
            
            q_values = self.q_networks[self.action_leader](embeddings)
            q_values = q_values.q_values.squeeze()
            action = torch.argmax(q_values)
            
        
        if return_q_values:
            return action, q_values
        return action
    
    
    