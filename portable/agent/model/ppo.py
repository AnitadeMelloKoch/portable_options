import numpy as np
import torch 
from torch import nn 
import os

import pfrl 
from pfrl.agents import PPO 
from pfrl.utils.recurrent import one_step_forward
from pfrl.utils.mode_of_distribution import mode_of_distribution
from pfrl import explorers
import gin

class MaskedPPO(PPO):
    def _batch_act_eval(self, batch_obs):
        assert not self.training
        b_state = self.batch_states(batch_obs, self.device, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)

        with torch.no_grad(), pfrl.utils.evaluating(self.model):
            if self.recurrent:
                (action_distrib, _), self.test_recurrent_states = one_step_forward(
                    self.model, b_state, self.test_recurrent_states
                )
            else:
                action_distrib, _ = self.model(b_state)
            if self.act_deterministically:
                print(mode_of_distribution(action_distrib).cpu().numpy())
                action = mode_of_distribution(action_distrib).cpu().numpy()
            else:
                print(action_distrib.sample().cpu().numpy())
                action = action_distrib.sample().cpu().numpy()

        return action
    
    def _batch_act_train(self, batch_obs):
        assert self.training
        b_state = self.batch_states(batch_obs, self.device, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)

        num_envs = len(batch_obs)
        if self.batch_last_episode is None:
            self._initialize_batch_variables(num_envs)
        assert len(self.batch_last_episode) == num_envs
        assert len(self.batch_last_state) == num_envs
        assert len(self.batch_last_action) == num_envs

        # action_distrib will be recomputed when computing gradients
        with torch.no_grad(), pfrl.utils.evaluating(self.model):
            if self.recurrent:
                assert self.train_prev_recurrent_states is None
                self.train_prev_recurrent_states = self.train_recurrent_states
                (
                    (action_distrib, batch_value),
                    self.train_recurrent_states,
                ) = one_step_forward(
                    self.model, b_state, self.train_prev_recurrent_states
                )
            else:
                action_distrib, batch_value = self.model(b_state)
            print(action_distrib.sample().cpu().numpy())
            batch_action = action_distrib.sample().cpu().numpy()
            self.entropy_record.extend(action_distrib.entropy().cpu().numpy())
            self.value_record.extend(batch_value.cpu().numpy())

        self.batch_last_state = list(batch_obs)
        self.batch_last_action = list(batch_action)

        return batch_action

def create_linear_policy(input_dim, action_space):
    return torch.nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.Tanh(),
                    nn.Linear(64, 64),
                    nn.Tanh(),
                    nn.Linear(64, action_space),
                    pfrl.policies.GaussianHeadWithStateIndependentCovariance(
                        action_size=action_space,
                        var_type="diagonal",
                        var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
                        var_param_init=0,  # log std = 0 => std = 1
                    ),
                )

def create_linear_vf(input_dim):
    return torch.nn.Sequential(
                            nn.Linear(input_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, 1),
                        )

@gin.configurable
class ActionPPO():
    def __init__(self,
                 use_gpu,
                 policy,
                 value_function,
                 learning_rate,
                 state_shape,
                 phi,
                 final_epsilon,
                 final_exploration_frames,
                 num_actions,
                 epochs_per_update=10,
                 clip_eps_vf=None,
                 entropy_coef=0,
                 standardize_advantages=True,
                 gamma=0.9,
                 lambd=0.97,
                 minibatch_size=64,
                 update_interval=2048):
        
        model = pfrl.nn.Branched(policy, value_function)
        opt = torch.optim.Adam(model.parameters(),
                               lr=learning_rate,
                               eps=1e-5)
        
        obs_normalizer = pfrl.nn.EmpiricalNormalization(state_shape,
                                                        clip_threshold=5)
        
        if use_gpu is False:
            gpu = -1
        else:
            gpu = 0
        
        self.agent = PPO(model,
                         opt,
                         obs_normalizer=obs_normalizer,
                         gpu=gpu,
                         phi=phi,
                         update_interval=update_interval,
                         minibatch_size=minibatch_size,
                         epochs=epochs_per_update,
                         clip_eps_vf=clip_eps_vf,
                         entropy_coef=entropy_coef,
                         standardize_advantages=standardize_advantages,
                         gamma=gamma,
                         lambd=lambd)
        
        self.explorer = explorers.LinearDecayEpsilonGreedy(1.0,
                                                           final_epsilon,
                                                           final_exploration_frames,
                                                           lambda: np.random.randint(num_actions))
        
        self.step = 0
    
    def save(self, dir):
        os.makedirs(dir, exist_ok=True)
        
        self.agent.save(dir)
    
    def load(self, dir):
        self.agent.load(dir)
    
    def act(self, obs):
        self.step += 1
        obs = obs.unsqueeze(0)
        q_vals = self.agent.batch_act(obs)
        if self.agent.training:
            action = self.explorer.select_action(
                self.step,
                greedy_action_func=lambda: torch.argmax(q_vals)
            )
        else:
            action = torch.argmax(q_vals)
        
        return action, q_vals
    
    def observe(self, obs, reward, done, reset):
        obs = obs.unsqueeze(0)
        reward = [reward]
        done = [done]
        reset = [reset]
        return self.agent.batch_observe(obs,
                                        reward,
                                        done,
                                        reset)
    

@gin.configurable
class OptionPPO():
    def __init__(self,
                 use_gpu,
                 policy,
                 value_function,
                 learning_rate,
                 state_dim,
                 action_num,
                 phi,
                 num_options,
                 final_epsilon,
                 final_exploration_frames,
                 epochs_per_update=20,
                 clip_eps_vf=None,
                 entropy_coef=0,
                 standardize_advantages=True,
                 gamma=0.9,
                 lambd=0.97,
                 minibatch_size=64,
                 update_interval=2048):
        
        model = pfrl.nn.Branched(policy, value_function)
        
        opt = torch.optim.Adam(model.parameters(),
                               lr=learning_rate,
                               eps=1e-5)
        
        obs_normalizer = pfrl.nn.EmpiricalNormalization((state_dim+action_num,))
        
        if use_gpu is False:
            gpu=-1
        else:
            gpu=0
        
        self.agent = PPO(model,
                         opt,
                         obs_normalizer=obs_normalizer,
                         gpu=gpu,
                         phi=phi,
                         update_interval=update_interval,
                         minibatch_size=minibatch_size,
                         epochs=epochs_per_update,
                         clip_eps_vf=clip_eps_vf,
                         entropy_coef=entropy_coef,
                         standardize_advantages=standardize_advantages,
                         gamma=gamma,
                         lambd=lambd)
        
        self.explorer = explorers.LinearDecayEpsilonGreedy(1.0,
                                                           final_epsilon,
                                                           final_exploration_frames,
                                                           lambda: np.random.randint(num_options))
        
        self.step = 0
    
    def save(self, dir):
        os.makedirs(dir, exist_ok=True)
        
        self.agent.save(dir)
    
    def load(self, dir):
        self.agent.load(dir)
    
    def act(self, obs, action):
        obs = obs.unsqueeze(0)
        action = torch.from_numpy(action)
        concat_input = torch.cat((obs, action), dim=-1)
        q_vals = self.agent.batch_act(concat_input)
        self.step += 1
        if self.agent.training:
            a = self.explorer.select_action(
                self.step,
                greedy_action_func=lambda: torch.argmax(q_vals)
            )
        else:
            a = torch.argmax(q_vals)
        
        return a
    
    def observe(self, obs, q_vals, reward, done, reset):
        obs = obs.unsqueeze(0)
        q_vals = torch.from_numpy(q_vals)
        concat_input = torch.cat((obs, q_vals), dim=-1)
        reward = [reward]
        done = [done]
        reset = [reset]
        return self.agent.batch_observe(concat_input,
                                        reward,
                                        done,
                                        reset)
        
        
        