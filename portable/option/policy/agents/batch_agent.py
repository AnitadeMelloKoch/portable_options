from typing import Any
import torch.nn as nn
import numpy as np 
from pfrl import replay_buffer
from pfrl.utils.batch_states import batch_states
import logging 
import gin 
import torch 
import itertools 
from contextlib import contextmanager

from portable.option.policy.agents import Agent, evaluating
from portable.option.policy import UpperConfidenceBound
from portable.option.ensemble.custom_attention import *
from portable.option.policy.models.ppo import PPO 
from portable.option.policy.models.ppo_cnn import ProcgenCNN

logger = logging.getLogger(__name__)

@gin.configurable 
class BatchedAgent(Agent):
    """
    Agent for PPO for procgen
    """
    def __init__(self,
                 learning_rate,
                 use_gpu,
                 warmup_steps,
                 batch_size,
                 phi,
                 buffer_length,
                 update_interval,
                 discount_rate,
                 num_actions,
                 ppo_lambda,
                 value_function_coef,
                 entropy_coef,
                 num_envs,
                 step_epochs,
                 clip_range,
                 max_grad_norm):
        super().__init__()
        
        self.use_gpu = use_gpu
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size
        self.phi = phi
        self.buffer_length = buffer_length
        self.update_interval = update_interval
        self.discount_rate = discount_rate
        self.num_actions = num_actions
        self.ppo_lambda = ppo_lambda
        self.value_function_coef = value_function_coef
        self.entropy_coef = entropy_coef
        self.num_envs = num_envs
        self.step_epochs = step_epochs
        self.clip_range = clip_range
        self.max_grad_norm = max_grad_norm
        
        self.step_number = 0
        self.episode_number = 0
        self.n_updates = 0
        
        self.agent = self.make_ppo_agent()
        
        self.optimizer = torch.optim.Adam(self.agent.model.parameters(),
                                          lr=self.learning_rate)
        
    def save(self, save_dir):
        pass
    
    def load(self, load_dir):
        pass 
    
    def make_ppo_agent(self):
        policy = ProcgenCNN(num_outputs=self.num_actions,
                            obs_shape=(3, 64, 64))
        agent = PPO(model=policy,
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
                    max_grad_norm=self.max_grad_norm)
        
        return agent
    
    @contextmanager
    def set_evaluating(self):
        istraining = self.training
        original_status = self.agent.training
        try:
            self.agent.training = istraining
            yield
        finally:
            self.agent.training = original_status
    
    def batch_observe(self,
                      batch_obs,
                      batch_reward,
                      batch_done,
                      batch_reset):
        with self.set_evaluating():
            loss = self.agent.batch_observe(batch_obs=batch_obs,
                                    batch_reward=batch_reward,
                                    batch_done=batch_done,
                                    batch_reset=batch_reset)
        
        if loss is not None:
            self.optimizer.zero_grad()
            loss.backward()
            
            nn.utils.clip_grad_norm(
                self.agent.model.parameters(),
                self.agent.max_grad_norm
            )
            self.optimizer.step()
        
        if self.training:
            self._batch_observe_train
        
    def _batch_observe_train(self,
                             batch_obs,
                             batch_reward,
                             batch_done,
                             batch_reset):
        for i in range(len(batch_obs)):
            if self.batch_last_obs[i] is not None:
                if batch_reset[i] or batch_done[i]:
                    self.batch_last_obs[i] = None
                    self.batch_last_action[i] = None
        
        if batch_reset.any() or batch_done.any():
            self.episode_number += np.logical_or(batch_reset, batch_done).sum()
    
    def _batch_observe_eval(self,
                            batch_obs,
                            batch_reward,
                            batch_done,
                            batch_reset):
        return
    
    def observe(self, 
                obs, 
                reward: float, 
                done: bool, 
                reset: bool) -> None:
        pass
    
    def batch_act(self,
                  batch_obs):
        with self.set_evaluating():
            batch_action = self.agent.batch_act((batch_obs))
            
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
