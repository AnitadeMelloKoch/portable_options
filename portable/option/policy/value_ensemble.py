import os
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from portable.option.ensemble.criterion import batched_L_divergence
from portable.option.ensemble.attention import Attention 
from portable.option.policy.models import LinearQFunction, compute_q_learning_loss
from portable.option.policy import UpperConfidenceBound


class ValueEnsemble():

    def __init__(self, 
        device,
        stack_size=4,
        embedding_output_size=64, 
        gru_hidden_size=128,
        learning_rate=2.5e-4,
        discount_rate=0.9,
        num_modules=8, 
        num_output_classes=18,
        plot_dir=None,
        verbose=False,
        c=100):
        
        self.num_modules = num_modules
        self.num_output_classes = num_output_classes
        self.device = device
        self.gamma = discount_rate
        self.verbose = verbose

        self.embedding = Attention(
            stack_size=stack_size,
            embedding_size=embedding_output_size, 
            num_attention_modules=self.num_modules
        ).to(self.device)

        self.recurrent_memory = nn.GRU(
            input_size=embedding_output_size,
            hidden_size=gru_hidden_size,
            batch_first=True,
        ).to(self.device)

        self.q_networks = nn.ModuleList(
            [LinearQFunction(in_features=gru_hidden_size, n_actions=num_output_classes) for _ in range(self.num_modules)]
        ).to(self.device)
        self.target_q_networks = deepcopy(self.q_networks)
        self.target_q_networks.eval()

        self.optimizer = optim.Adam(
            list(self.embedding.parameters()) + list(self.q_networks.parameters()) + list(self.recurrent_memory.parameters()),
            learning_rate,
        )

        self.upper_confidence_bound = UpperConfidenceBound(num_modules=num_modules, c=c, device=device)
        self.action_leader = np.random.choice(num_modules)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.embedding.state_dict(), os.path.join(path, 'embedding.pt'))
        torch.save(self.q_networks.state_dict(), os.path.join(path, 'policy_networks.pt'))
        self.upper_confidence_bound.save(os.path.join(path, 'upper_conf_bound'))

    def load(self, path):
        self.embedding.load_state_dict(torch.load(os.path.join(path, 'embedding.pt')))
        self.q_networks.load_state_dict(torch.load(os.path.join(path, 'policy_networks.pt')))
        self.upper_confidence_bound.load(os.path.join(path, 'upper_conf_bound'))

    def step(self):
        self.upper_confidence_bound.step()

    def update_accumulated_rewards(self, reward):
        self.upper_confidence_bound.update_accumulated_rewards(self.action_leader, reward)

    def update_leader(self):
        self.action_leader = self.upper_confidence_bound.select_leader()

    def train(self, exp_batch, errors_out=None, update_target_network=False):
        """
        update both the embedding network and the value network by backproping
        the sumed divergence and q learning loss
        """
        self.embedding.train()
        self.q_networks.train()
        self.recurrent_memory.flatten_parameters()

        batch_states = exp_batch['state']
        batch_actions = exp_batch['action']
        batch_rewards = exp_batch['reward']
        batch_next_states = exp_batch['next_state']
        batch_dones = exp_batch['is_state_terminal']

        loss = 0

        # divergence loss
        state_embeddings = self.embedding(batch_states, return_attention_mask=False)  # (batch_size, num_modules, embedding_size)
        state_embeddings, _ = self.recurrent_memory(state_embeddings)  # (batch_size, num_modules, gru_out_size)
        l_div = batched_L_divergence(state_embeddings, self.upper_confidence_bound.weights())
        loss += l_div

        # q learning loss
        td_losses = np.zeros((self.num_modules,))
        next_state_embeddings = self.embedding(batch_next_states, return_attention_mask=False)
        next_state_embeddings, _ = self.recurrent_memory(next_state_embeddings)

        # keep track of all error out for each module 
        all_errors_out = np.zeros((self.num_modules, len(batch_states)))

        for idx in range(self.num_modules):

            # predicted q values
            state_attention = state_embeddings[:,idx,:]  # (batch_size, emb_out_size)
            batch_pred_q_all_actions = self.q_networks[idx](state_attention)  # (batch_size, num_actions)
            batch_pred_q = batch_pred_q_all_actions.evaluate_actions(batch_actions)  # (batch_size,)

            # target q values 
            with torch.no_grad():
                next_state_attention = next_state_embeddings[:,idx,:]  # (batch_size, emb_out_size)
                batch_next_state_q_all_actions = self.target_q_networks[idx](next_state_attention)  # (batch_size, num_actions)
                next_state_values = batch_next_state_q_all_actions.max  # (batch_size,)
                batch_q_target = batch_rewards + self.gamma * (1-batch_dones) *  next_state_values # (batch_size,)
            
            # loss
            td_loss = compute_q_learning_loss(exp_batch, batch_pred_q, batch_q_target, errors_out=errors_out)
            all_errors_out[idx] = errors_out
            loss += td_loss
            if self.verbose: td_losses[idx] = td_loss.item()

        # update errors_out, so it accounts for all modules in ensemble
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
        if self.verbose:
            # for idx in range(self.num_modules):
            #     print("\t - Value {}: loss {:.6f}".format(idx, td_losses[idx]))
            print(f"Div loss: {l_div.item()}. Q loss: {np.sum(td_losses)}")

        self.embedding.eval()
        self.q_networks.eval()

    def predict_actions(self, state, return_q_values=False):
        """
        given a state, each one in the ensemble predicts an action
        args:
            return_q_values: if True, return the predicted q values each learner predicts on the action of their choice.
        """
        self.embedding.eval()
        self.q_networks.eval()
        with torch.no_grad():
            embeddings = self.embedding(state, return_attention_mask=False).detach()
            self.recurrent_memory.flatten_parameters()
            embeddings, _ = self.recurrent_memory(embeddings)

            actions = np.zeros(self.num_modules, dtype=np.int)
            q_values = np.zeros(self.num_modules, dtype=np.float)
            all_q_values = np.zeros((self.num_modules, self.num_output_classes), dtype=np.float)
            for idx in range(self.num_modules):
                attention = embeddings[:,idx,:]
                q_vals = self.q_networks[idx](attention)
                actions[idx] = q_vals.greedy_actions
                q_values[idx] = q_vals.max
                all_q_values[idx, :] = q_vals.q_values.cpu().numpy()

            selected_action = actions[self.action_leader]

        if return_q_values:
            return selected_action, actions, q_values, all_q_values
        return selected_action

    def get_single_module(self, state, module):
        self.embedding.eval()
        self.q_networks.eval()
        
        embedding = self.embedding.forward_one_attention(state, module).squeeze()
        embedding, _ = self.recurrent_memory(embedding)
        
        qvals = self.q_networks[module](embedding)
        
        return qvals
            

    def get_attention(self, x):
        self.embedding.eval()
        x = x.to(self.device)
        _, atts = self.embedding(x, return_attention_mask=True).detach()
        return atts
