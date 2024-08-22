from pfrl.utils.batch_states import batch_states
import itertools 
from pfrl.agents import PPO
import numpy as np
import torch 
from torch import nn 
import os
import pfrl
import logging
import gin
from collections import deque

import matplotlib.pyplot as plt

# def create_minigrid_model(n_channels=3, action_space=7):
#     return nn.Sequential(
#         nn.Conv2d(n_channels, 16, (2,2)),
#         nn.BatchNorm2d(16),
#         nn.ReLU(),
#         nn.Conv2d(16, 32, (2,2)),
#         nn.BatchNorm2d(32),
#         nn.ReLU(),
#         nn.Conv2d(32, 64, (2,2)),
#         nn.BatchNorm2d(64),
#         nn.ReLU(),
#         nn.Flatten(),
#         nn.LazyLinear(512),
#         nn.ReLU(),
#         pfrl.nn.Branched(
#             nn.Sequential(
#                 nn.Linear(512, 64),
#                 nn.Tanh(),
#                 nn.Linear(64, action_space),
#                 pfrl.policies.SoftmaxCategoricalHead()
#             ),
#             nn.Sequential(
#                 nn.Linear(512, 64),
#                 nn.Tanh(),
#                 nn.Linear(64, 1)
#             )
#         )
#     )

class PrintLayer(torch.nn.Module):
    # print input. For debugging
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        print(x.shape)
        
        return x

class VisLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        
        fig = plt.figure(num=1, clear=True)
        ax = fig.add_subplot()
        ax.imshow(x.cpu().numpy()[0,0,:,:])
        plt.show(block=False)
        input("inside model")
    
        return x

def create_minigrid_model(n_channels=3, action_space=7):
    return nn.Sequential(
        nn.Conv2d(n_channels, 32, 8, stride=2),
        # PrintLayer(),
        # VisLayer(),
        nn.ReLU(),
        nn.Conv2d(32,64,3,stride=2),
        # PrintLayer(),
        # VisLayer(),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        # PrintLayer(),
        # VisLayer(),
        nn.Flatten(),
        nn.LazyLinear(512),
        nn.ReLU(),
        pfrl.nn.Branched(
            nn.Sequential(
                nn.Linear(512, action_space),
                pfrl.policies.SoftmaxCategoricalHead(),
            ),
            nn.Sequential(
                nn.Linear(512, 1),
            )
        )
    )


@gin.configurable
class SkillPPO():
    def __init__(self,
                 use_gpu,
                 learning_rate,
                 state_shape,
                 phi,
                 num_actions,
                 model_type="minigrid",
                 epochs_per_update=10,
                 clip_eps_vf=None,
                 entropy_coef=0,
                 standardize_advantages=True,
                 gamma=0.9,
                 lambd=0.97,
                 minibatch_size=64,
                 update_interval=2048):
        
        if model_type == "minigrid":
            model = create_minigrid_model()
        
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
        
        self.step_number = 0
        self.train_rewards = deque(maxlen=200)
        self.option_runs = 0
    
    def save(self, dir):
        os.makedirs(dir, exist_ok=True)
        
        self.agent.save(dir)
    
    def load(self, dir):
        print("\033[92m {}\033[00m" .format("PPO model loaded"))
        self.agent.load(dir)
    
    def act(self, obs, return_q=False):
        out = self.agent.batch_act([obs])
        out = torch.from_numpy(out)
        
        # action = torch.argmax(out, axis=-1)
        
        if return_q is True:
            return out, out
        else:
            return out
    
    def q_function(self, obs):
        return self.agent.batch_act(obs)
    
    def observe(self, obs, action, reward, next_obs, terminal):
        self.update_step()
        if type(obs) == np.ndarray:
            obs = torch.from_numpy(obs)
        obs = obs.unsqueeze(0)
        reward = [reward]
        done = [terminal]
        reset = [terminal]
        
        self.agent.batch_observe(obs,
                                 reward,
                                 done,
                                 reset)
    
    def update_step(self):
        self.step_number += 1
    
    def end_skill(self, summed_reward):
        self.train_rewards.append(summed_reward)
        self.option_runs += 1
        if self.option_runs%50 == 0:
            logging.info("Option policy success rate: {} from {} episodes {} steps".format(np.mean(self.train_rewards), 
                                                                                           self.option_runs,
                                                                                           self.step_number))
    

