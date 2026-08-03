import logging
import os
import numpy as np
import gin
import random
import torch
import pickle
from copy import deepcopy

from portable.option.vf_transfer.policy.uncertain_ppo import PPOEnsembleVFAgent
from portable.option.vf_transfer.models.uncertain_ppo_model import VFEnsembleFull

@gin.configurable
class PPOSkillLibrary():
    def __init__(self,
                 num_meta_vf_heads,
                 phi=lambda x: x):
        self.skills = []
        self.meta_vf = VFEnsembleFull(num_meta_vf_heads)
        self.meta_vf_ready = False
        self.phi = phi

    def add_skill(self):
        agent = PPOEnsembleVFAgent(meta_vf=self.meta_vf,
                                   phi=self.phi)
        if self.meta_vf_ready:
            agent.agent.meta_vf_ready = True
        self.skills.append(agent)

    def check_init(self, state):
        if len(self.skills) == 1:
            return self.skills[0]

    def run_task(self, env):
        skill = self.check_init(None)
        return skill.run_episode(env)

    def consolidate_v(self):
        task_model = self.skills[-1].agent.model
        self.meta_vf.heads = deepcopy(task_model.vf_ensemble.heads)
        for param in self.meta_vf.parameters():
            param.requires_grad = False
        self.meta_vf.eval()
        self.meta_vf_ready = True

    def __len__(self):
        return len(self.skills)
    
    
    
    



