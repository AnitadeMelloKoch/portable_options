import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from captum.attr import IntegratedGradients, NoiseTunnel
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
from ..evaluators.utils import concatenate
from evaluation.model_wrappers import EnsemblePolicyWrapper
from portable.utils import set_player_ram

class AttentionEvaluatorPolicy():
    
    def __init__(self,
                 plot_dir,
                 policy,
                 env):
        self.device = torch.device("cuda")
        
        self.policy = policy
        
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)
        self.num_modules = self.policy.num_modules
        self.env = env
        
        self.integrated_gradients = []
        for x in range(self.num_modules):
            self.integrated_gradients.append(NoiseTunnel(
                IntegratedGradients(
                    EnsemblePolicyWrapper(
                        self.policy,
                        x
                    )
                )
            ))
        
    def _set_env_ram(self, ram, state, agent_state):
        self.env.reset()
        _ = set_player_ram(self.env, ram)
        self.env.stacked_state = state
        self.env.stacked_agent_state = agent_state

        return state
        
    def evaluate(self, starting_ram):
        
        state = self._set_env_ram(
            starting_ram["ram"],
            starting_ram["state"],
            starting_ram["agent_state"]
        )
        steps = 0
        while steps < 15:
            
            action = self.policy.act(state)
            
            steps += 1

            state, reward, done, info = self.env.step(action)
            
            attributions = self._attributions(state, self.integrated_gradients)
            
            action = self.policy.act(state)
            
    def _attributions(self, state, integrated_gradients):
        attributions = []
        for image in state:
            single_image = image.unsqueeze(0)
            image_attr = []
            for ig in integrated_gradients:
                image_attr.append(ig.attribute(
                    single_image,
                    nt_samples=10,
                    n_steps=10
                ))
            attributions.append(image_attr)
            
            
            
            