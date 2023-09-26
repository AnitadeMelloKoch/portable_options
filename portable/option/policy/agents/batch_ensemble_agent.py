import torch.nn as nn
import numpy as np 
from pfrl import replay_buffers
from pfrl.replay_buffer import batch_experiences, ReplayUpdater
from pfrl.utils.batch_states import batch_states
import logging
import gin
import torch
from contextlib import contextmanager
import itertools

from portable.option.policy.agents import Agent, evaluating
from portable.option.policy import UpperConfidenceBound
from portable.option.ensemble.custom_attention import *
from portable.option.policy.models.ppo import PPO
from portable.option.policy.models.ppo_mlp import PPOMLP

logger = logging.getLogger(__name__)

@gin.configurable
class BatchedEnsembleAgent(Agent):
    """
    Agent takes an ensemble of policies.
    Supports batch_act and batch_observe
    """
    def __init__(self,
                 embedding: AutoEncoder,
                 embedding_phi,
                 learning_rate,
                 num_modules,
                 use_gpu,
                 warmup_steps,
                 batch_size,
                 phi,
                 buffer_length,
                 update_interval,
                 discount_rate,
                 bandit_exploration_weight,
                 num_actions,
                 ppo_lambda,
                 value_function_coef,
                 entropy_coef,
                 num_envs,
                 step_epochs,
                 clip_range,
                 max_grad_norm,
                 divergence_loss_scale,
                 fix_attention=False):
        super().__init__()
        
        self.use_gpu = use_gpu
        self.embedding = embedding
        self.embedding_phi = embedding_phi
        self.learning_rate = learning_rate
        self.num_modules = num_modules
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size
        self.phi = phi
        self.buffer_length = buffer_length
        self.update_interval = update_interval
        self.discount_rate = discount_rate
        self.bandit_exploration_weight = bandit_exploration_weight
        self.fix_attention = fix_attention
        self.num_actions = num_actions
        self.ppo_lambda = ppo_lambda
        self.value_function_coef = value_function_coef
        self.entropy_coef = entropy_coef
        self.num_envs = num_envs
        self.step_epochs = step_epochs
        self.clip_range = clip_range
        self.max_grad_norm = max_grad_norm
        self.divergence_loss_scale = divergence_loss_scale
        
        self.action_leader = np.random.choice(num_modules)
        
        self.step_number = 0
        self.episode_number = 0
        self.n_updates = 0
        
        self.head_accumulated_reward = np.zeros(num_modules)
        self.head_selection_count = np.zeros(num_modules)
        
        self.upper_confidence_bound = UpperConfidenceBound(num_modules=num_modules,
                                                           c = self.bandit_exploration_weight)
        
        
        
        # attention layer
        self.attentions = nn.ModuleList(
            [AttentionLayer(self.embedding.feature_size) for _ in range(self.num_modules)]
        )
        if self.use_gpu:
            self.attentions = self.attentions.to("cuda")
        # ppo agent
        self.heads = [self.make_ppo_agent() for _ in range(self.num_modules)]
        
        learnable_parameters = list(itertools.chain.from_iterable(
            [list(head.model.parameters()) for head in self.heads])
        )
        learnable_parameters = list(self.attentions.parameters()) + learnable_parameters
        
        self.optimizer = torch.optim.Adam(learnable_parameters,
                                          lr=self.learning_rate)

        self.optimizers = [
            torch.optim.Adam(
                params=list(self.attentions[idx].parameters())+list(self.heads[idx].model.parameters()),
                lr=learning_rate
            ) for idx in range(self.num_modules)
        ]
        
        
    def save(self, save_dir):
        pass
    
    def load(self, load_dir):
        pass
    
    def make_ppo_agent(self):
        policy = PPOMLP(output_size=self.num_actions)
        agent = PPO(
            model=policy,
            optimizer=None,
            gpu=0 if self.use_gpu else -1,
            gamma=self.discount_rate,
            lambd=self.ppo_lambda,
            phi=self.phi,
            value_func_coef=self.value_function_coef,
            entropy_coef=self.entropy_coef,
            update_interval=self.n_updates*self.num_envs,
            minibatch_size=self.batch_size,
            epochs=self.step_epochs,
            clip_eps=self.clip_range,
            clip_eps_vf=self.clip_range,
            max_grad_norm=self.max_grad_norm
        )
        
        return agent
    
    @contextmanager
    def set_evaluating(self):
        istraining = self.training
        original_status = [head.training for head in self.heads]
        try:
            for head in self.heads:
                head.training = istraining
            yield
        finally:
            for head, status in zip(self.heads, original_status):
                head.training = status
    
    def batch_observe(self,
                      batch_obs,
                      batch_reward,
                      batch_done,
                      batch_reset):
        with self.set_evaluating():
            embedded_obs = self._attention_embed_obs(batch_obs, self.action_leader)
            
            loss = self.heads[self.action_leader].batch_observe(embedded_obs,
                                                                batch_reward,
                                                                batch_done,
                                                                batch_reset)
        
        if loss is not None:
            masks = self.get_attention_masks()
            div_loss = self.divergence_loss_scale*divergence_loss(masks, self.action_leader)

            loss += div_loss
            
            self.attentions[self.action_leader].train()
            self.optimizers[self.action_leader].zero_grad()
            loss.backward()
            
            nn.utils.clip_grad_norm(
                self.heads[self.action_leader].model.parameters(),
                self.heads[self.action_leader].max_grad_norm
            )
            self.optimizers[self.action_leader].step()
            self.attentions[self.action_leader].eval()
        
        if self.training:
            self._batch_observe_train(batch_obs,
                                      batch_reward,
                                      batch_done,
                                      batch_reset)
        else:
            self._batch_observe_eval(batch_obs,
                                     batch_reward,
                                     batch_done,
                                     batch_reset)
        
        # if np.sum([loss is not None for loss in losses]) == self.num_modules:
        #     head_loss = torch.stack(losses).mean()
            
        #     masks = self.get_attention_masks()
        #     div_loss = 0
        #     for idx in range(self.num_modules):
        #         div_loss += divergence_loss(masks, idx)
            
        #     div_loss = self.divergence_loss_scale*div_loss
            
        #     loss = head_loss + div_loss
            
        #     self.attentions.train()
        #     self.optimizer.zero_grad()
        #     loss.backward()
        #     for head in self.heads:
        #         if hasattr(head, 'max_grad_norm') and head.max_grad_norm is not None:
        #             nn.utils.clip_grad_norm(head.model.parameters(), head.max_grad_norm)
        #     self.optimizer.step()
        #     self.attentions.eval()
        
    
    def get_attention_masks(self):
        masks = []
        for attention in self.attentions:
            masks.append(attention.mask())
        
        return masks
    
    def _batch_observe_train(self,
                             batch_obs,
                             batch_reward,
                             batch_done,
                             batch_reset):
        for i in range(len(batch_obs)):
            self.upper_confidence_bound.step()
            
            if self.batch_last_obs[i] is not None:
                if batch_reset[i] or batch_done[i]:
                    self.batch_last_obs[i] = None
                    self.batch_last_action[i] = None

        if batch_reset.any() or batch_done.any():
            self.episode_number += np.logical_or(batch_reset, batch_done).sum()
            self._set_action_leader()
    
    
    def _batch_observe_eval(self,
                            batch_obs,
                            batch_reward,
                            batch_done,
                            batch_reset):
        return
    
    def _update_learner_stats(self, reward):
        self.upper_confidence_bound.update_accumulated_rewards(self.action_leader,
                                                               reward)
    
    def _attention_embed_obs(self, batch_obs, idx):
        batch_obs = self.embedding_phi(batch_obs, self.use_gpu)
        obs_embeddings = self.embedding.feature_extractor(batch_obs)
        
        attentioned_embedding = self.attentions[idx](obs_embeddings)
        
        if self.use_gpu:
            return attentioned_embedding.to("cuda")
        else:
            return attentioned_embedding
    
    def observe(self,
                obs,
                action,
                reward,
                next_obs,
                terminal):
        pass
    
    def _set_action_leader(self):
        self.action_leader = self.upper_confidence_bound.select_leader()
    
    def batch_act(self, batch_obs):
        with self.set_evaluating():
            # learners choose actions
            embedded_obs = self._attention_embed_obs(batch_obs, self.action_leader)
            batch_action = self.heads[self.action_leader].batch_act((embedded_obs))

            self.batch_last_obs = list(batch_obs)
            self.batch_last_action = list(batch_action)

            return batch_action
    
    def act(self, obs):
        """
        epsilon-greedy policy
        args:
            obs (object): Observation from the environment.
            return_ensemble_info (bool): when set to true, this function returns
                (action_selected, actions_selected_by_each_learner, q_values_of_each_actions_selected)
        """
        with torch.no_grad(), evaluating(self):
            obs = batch_states([obs], self.device, self.phi)
            actions, action_q_vals, all_q_vals = self.value_ensemble.predict_actions(obs, return_q_values=True)
            actions, action_q_vals, all_q_vals = actions[0], action_q_vals[0], all_q_vals[0]  # get rid of batch dimension
        # action selection strategy
        action_selection_func = lambda a, qvals: a[self.action_leader]
        # epsilon-greedy
        if self.training:
            a = self.explorer.select_action(
                self.step_number,
                greedy_action_func=lambda: action_selection_func(actions, all_q_vals),
            )
        else:
            a = actions[self.action_leader]
        return a
    
    

