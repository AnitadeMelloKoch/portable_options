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
from typing import Any, Callable, Sequence
from torch.utils.data._utils.collate import default_collate
from pfrl.policies import SoftmaxCategoricalHead

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
        if self.batch_last_state[0] is not None:
            assert len(self.batch_last_state[0]) == num_envs
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
            
            batch_action = action_distrib.sample().cpu().numpy()
            self.entropy_record.extend(action_distrib.entropy().cpu().numpy())
            self.value_record.extend(batch_value.cpu().numpy())

        self.batch_last_state = list(batch_obs)
        self.batch_last_action = list(batch_action)

        return batch_action

def batch_states(states:Sequence[Any], 
                 device: torch.device, 
                 phi: Callable[[Any], Any]):
    features = [phi(s[0]) for s in states]
    masks = [s[1] for s in states]
    masks = torch.tensor(masks)
    masks = masks.squeeze(1)
    collated_features = default_collate(features)
    collated_features = collated_features.squeeze(1)
    collated_features = collated_features.to(device)
    
    return (collated_features, masks)

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

def create_cnn_policy(n_channels, action_space, hidden_feature_size=128):
    return torch.nn.Sequential(
        nn.Conv2d(n_channels, 16, (2,2)),
        nn.ReLU(),
        nn.Conv2d(16, 32, (2,2)),
        nn.ReLU(),
        nn.Conv2d(32, 64, (2,2)),
        nn.ReLU(),
        nn.Flatten(),
        nn.LazyLinear(hidden_feature_size),
        nn.ReLU(),
        
        nn.Linear(hidden_feature_size, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, action_space),
        pfrl.policies.GaussianHeadWithStateIndependentCovariance(
                        action_size=action_space,
                        var_type="diagonal",
                        var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
                        var_param_init=0,  # log std = 0 => std = 1
                    )
    )

class CNNPolicy(nn.Module):
    def __init__(self,
                 n_channels,
                 action_space,
                 hidden_feature_size=128):
        super().__init__()
        
        self.model = torch.nn.Sequential(
            nn.Conv2d(n_channels, 16, (2,2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2,2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2,2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(hidden_feature_size),
            nn.ReLU(),
            
            nn.Linear(hidden_feature_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_space),
        )
    
    def forward(self, x):
        action_vals = self.model(x)
        
        return action_vals

class CNNVF(nn.Module):
    def __init__(self, n_channels, hidden_feature_size=128):
        super().__init__()
        self.model = torch.nn.Sequential(
            nn.Conv2d(n_channels, 16, (2,2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2,2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2,2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(hidden_feature_size),
            nn.ReLU(),
            
            nn.Linear(hidden_feature_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        
        return self.model(x)

class MaskedEmpiricalNormalization(pfrl.nn.EmpiricalNormalization):
    def forward(self, x, update=True):
        if isinstance(x, tuple):
            obs = x[0]
            mask = x[1]
        else:
            obs = [s[0] for s in x]
            mask = [s[1] for s in x]
        normalized_x = super().forward(obs, update=update)
        
        return (normalized_x, mask)
    
    def experience(self, x):
        if isinstance(x, tuple):
            obs = x[0]
        else:
            obs = [s[0] for s in x]
            obs = torch.stack(obs, dim=0)
        return super().experience(obs)

def create_linear_vf(input_dim):
    return torch.nn.Sequential(
                            nn.Linear(input_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, 1),
                        )

def create_cnn_vf(n_channels, hidden_feature_size=128):
    return torch.nn.Sequential(
        nn.Conv2d(n_channels, 16, (2,2)),
        nn.ReLU(),
        nn.Conv2d(16, 32, (2,2)),
        nn.ReLU(),
        nn.Conv2d(32, 64, (2,2)),
        nn.ReLU(),
        nn.Flatten(),
        nn.LazyLinear(hidden_feature_size),
        nn.ReLU(),
        
        nn.Linear(hidden_feature_size, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1)
    )

def create_linear_atari_model(in_features, n_actions):
    return torch.nn.Sequential(
                            lecun_init(nn.Linear(in_features, 512)),
                            nn.ReLU(),
                            lecun_init(nn.Linear(512, 256)),
                            nn.ReLU(),
                            pfrl.nn.Branched(
                                nn.Sequential(
                                    lecun_init(nn.Linear(256, n_actions), 1e-2),
                                    pfrl.policies.GaussianHeadWithStateIndependentCovariance(
                                            action_size=n_actions,
                                            var_type="diagonal",
                                            var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
                                            var_param_init=0,  # log std = 0 => std = 1
                                        )
                                ),
                                lecun_init(nn.Linear(256, 1))
                            )
                        )

def lecun_init(layer, gain=1):
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        pfrl.initializers.init_lecun_normal(layer.weight, gain)
        nn.init.zeros_(layer.bias)
    else:
        pfrl.initializers.init_lecun_normal(layer.weight_ih_l0, gain)
        pfrl.initializers.init_lecun_normal(layer.weight_hh_l0, gain)
        nn.init.zeros_(layer.bias_ih_l0)
        nn.init.zeros_(layer.bias_hh_l0)
    return layer

class PrintLayer(torch.nn.Module):
    # print input. For debugging
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        print(x)
        
        return x

def create_atari_model(n_channels, n_actions):
    return nn.Sequential(
        lecun_init(nn.Conv2d(n_channels, 32, 8, stride=4)),
        nn.ReLU(),
        lecun_init(nn.Conv2d(32, 64, 4, stride=2)),
        nn.ReLU(),
        lecun_init(nn.Conv2d(64, 64, 3, stride=1)),
        nn.ReLU(),
        nn.Flatten(),
        lecun_init(nn.Linear(3136,512)),
        nn.ReLU(),
        pfrl.nn.Branched(
            nn.Sequential(
                lecun_init(nn.Linear(512, n_actions), 1e-2),
                pfrl.policies.GaussianHeadWithStateIndependentCovariance(
                        action_size=n_actions,
                        var_type="diagonal",
                        var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
                        var_param_init=0,  # log std = 0 => std = 1
                    )
            ),
            lecun_init(nn.Linear(512, 1))
        )
    )

@gin.configurable
class ActionPPO():
    def __init__(self,
                 use_gpu,
                 learning_rate,
                 state_shape,
                 phi,
                 num_actions,
                 policy=None,
                 value_function=None,
                 model=None,
                 epochs_per_update=10,
                 clip_eps_vf=None,
                 entropy_coef=0,
                 standardize_advantages=True,
                 gamma=0.9,
                 lambd=0.97,
                 minibatch_size=64,
                 update_interval=2048):
        
        self.returns_vals = True
        if model is None:
            assert policy is not None
            assert value_function is not None
            model = pfrl.nn.Branched(policy, value_function)
            self.returns_vals = True
        opt = torch.optim.Adam(model.parameters(),
                               lr=learning_rate,
                               eps=1e-5)
        
        obs_normalizer = pfrl.nn.EmpiricalNormalization(state_shape,
                                                        clip_threshold=5)
        
        self.agent = PPO(model,
                         opt,
                         obs_normalizer=obs_normalizer,
                         gpu=use_gpu,
                         phi=phi,
                         entropy_coef=entropy_coef,
                         update_interval=update_interval,
                         minibatch_size=minibatch_size,
                         epochs=epochs_per_update,
                         clip_eps_vf=clip_eps_vf,
                         max_grad_norm=1,
                         standardize_advantages=standardize_advantages,
                         gamma=gamma,
                         lambd=lambd)
        
        self.num_actions = num_actions
        
        self.step = 0
    
    def save(self, dir):
        os.makedirs(dir, exist_ok=True)
        
        self.agent.save(dir)
    
    def load(self, dir):
        print("\033[92m {}\033[00m" .format("PPO model loaded"))
        self.agent.load(dir)
    
    def act(self, obs, mask=None):
        self.step += 1
        out = self.agent.batch_act([obs])
        out = torch.from_numpy(out)
        
        if not self.returns_vals:
            return out, None
        
        if mask is not None:
            mask = torch.stack(mask).unsqueeze(0)
            out[mask] = torch.min(out)
        
        if self.agent.training:
            action = torch.argmax(out)
        
        return action, out
    
    def q_function(self, obs):
        return self.agent.batch_act(obs)
    
    def observe(self, obs, reward, done, reset):
        obs = obs.unsqueeze(0)
        reward = [reward]
        done = [done]
        reset = [reset]
        
        self.agent.batch_observe(obs,
                                 reward,
                                 done,
                                 reset)
    
