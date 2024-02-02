import logging 
import os 
import numpy as np 
import gin 
import random 
import torch 

from portable.option.divdis.divdis_classifier import DivDisClassifier
from portable.option.divdis.policy.policy_and_initiation import PolicyWithInitiation
from portable.option.policy.agents import evaluating
import matplotlib.pyplot as plt 

@gin.configurable 
class DivDisOption():
    def __init__(self,
                 use_gpu,
                 log_dir,
                 save_dir,
                 num_heads,
                 
                 policy_phi,
                 video_generator=None,
                 plot_dir=None):
        
        self.use_gpu = use_gpu
        self.save_dir = save_dir
        self.policy_phi = policy_phi
        self.log_dir = log_dir
        
        self.terminations = DivDisClassifier(use_gpu=use_gpu,
                                             head_num=num_heads,
                                             log_dir=os.path.join(log_dir, 'termination'))
        
        self.num_heads = num_heads
        
        self.policies = [
            {} for _ in range(num_heads)
        ]
        
        self.initiable_policies = None
        self.video_generator = video_generator
        self.make_plots = False
        
        if plot_dir is not None:
            self.make_plots = True
            self.plot_dir = plot_dir
            self.term_states = []
    
    def _video_log(self, line):
        if self.video_generator is not None:
            self.video_generator.add_line(line)
    
    def _get_termination_save_path(self):
        return os.path.join(self.save_dir, 'termination')
    
    def save(self):
        os.makedirs(self._get_termination_save_path(), exist_ok=True)
        
        self.terminations.save(path=self._get_termination_save_path())
    
    def load(self):
        if os.path.exists(self._get_termination_save_path()):
            # print in green text
            print("\033[92m {}\033[00m" .format("Termination model loaded"))
            self.terminations.load(self._get_termination_save_path())
        else:
            # print in red text
            print("\033[91m {}\033[00m" .format("No Checkpoint found. No model has been loaded"))
    
    def add_policy(self, 
                   term_idx):
        self.policies[term_idx].append(PolicyWithInitiation())
    
    def find_possible_policy(self, obs):
        policy_idxs = []
        
        for policies in self.policies:
            idxs = []
            for idx in range(len(policies)):
                if policies[idx].can_initiate(obs):
                    idxs.append(idx)
            policy_idxs.append(idxs)
        
        self.initiable_policies = policy_idxs
        
        return policy_idxs
    
    def add_datafiles(self,
                      positive_files,
                      negative_files,
                      unlabelled_files):
        self.terminations.add_data(positive_files,
                                   negative_files,
                                   unlabelled_files)
    
    def train_policy(self, 
                     idx,
                     env,
                     state,
                     info,
                     seed):
        
        steps = 0
        rewards = []
        option_rewards = []
        states = []
        infos = []
        
        done = False
        should_terminate = False
        
        # if seed not in self.initiable_policies[idx]:
        #     self.initiable_policies[idx][seed] = PolicyWithInitiation()
        
        # policy = self.initiable_policies[idx][seed]
        
        if seed not in self.policies[idx].keys():
            self.policies[idx][seed] = PolicyWithInitiation(use_gpu=self.use_gpu,
                                                            policy_phi=self.policy_phi)
        
        policy = self.policies[idx][seed]
        policy.move_to_gpu()
        
        img_state = None
        img_next_state = env.render()
        
        while not (done or should_terminate):
            states.append(state)
            infos.append(info)
            
            action = policy.act(state)
            
            next_state, reward, done, info = env.step(action)
            img_state = img_next_state
            img_next_state = env.render()
            term_state = self.policy_phi(next_state).unsqueeze(0)
            should_terminate = torch.argmax(self.terminations.predict_idx(term_state, idx)) == 1
            steps += 1
            rewards.append(reward)
            
            if should_terminate:
                reward = 1
                if self.make_plots:
                    np_next_state = list(next_state.cpu().numpy())
                    if np_next_state not in self.term_states:
                        self.plot_term_state(img_state, img_next_state, idx)
                        self.term_states.append(np_next_state)
            else:
                reward = 0
            
            policy.observe(state,
                           action,
                           reward,
                           next_state,
                           done or should_terminate)
            
            option_rewards.append(reward)

            state = next_state
        
        # if should_terminate:
        #     # should save positive examples and create classifier if it hasn't been
        #     pass
        # if done:
        #     pass
        
        return state, info, steps, rewards, option_rewards, states, infos
    
    def plot_term_state(self, state, next_state, idx):
        x = 0
        plot_dir = os.path.join(self.plot_dir, str(idx))
        os.makedirs(plot_dir, exist_ok=True)
        while os.path.exists(os.path.join(plot_dir, "{}.png".format(x))):
            x += 1
        plot_file = os.path.join(plot_dir, "{}.png".format(x))
        
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.imshow(state)
        ax1.axis('off')
        ax2.imshow(next_state)
        ax2.axis('off')
        
        fig.savefig(plot_file)
        plt.close(fig)
        
    
    def env_train_policy(self,
                         idx,
                         env,
                         state,
                         info,
                         seed):
        # train policy from environment rewards not termination function
        steps = 0
        rewards = []
        option_rewards = []
        states = []
        infos = []
        
        done = False
        
        if seed not in self.policies[idx].keys():
            self.policies[idx][seed] = PolicyWithInitiation(use_gpu=self.use_gpu,
                                                            policy_phi=self.policy_phi)
        
        policy = self.policies[idx][seed]
        policy.move_to_gpu()
        
        while not done:
            states.append(state)
            infos.append(info)
            
            action = policy.act(state)
            
            next_state, reward, done, info = env.step(action)
            steps += 1
            rewards.append(reward)
            policy.observe(state,
                           action,
                           reward,
                           next_state,
                           done)
            
            option_rewards.append(reward)
            
            state = next_state
        
        self.policies[idx][seed] = policy
        
        return state, info, steps, rewards, option_rewards, states, infos
    
    def eval_policy(self,
                    idx,
                    env,
                    state,
                    info,
                    seed):
        
        steps = 0
        rewards = []
        option_rewards = []
        states = []
        infos = []
        
        done = False
        should_terminate = False
        
        if seed not in self.policies[idx]:
            raise Exception("Policy has not been initialized. Train policy before evaluating")
        
        policy = self.policies[idx][seed]
        
        with evaluating(policy):
            while not (done or should_terminate):
                states.append(state)
                infos.append(info)
                
                action = policy.act(state)
                self._video_log("action: {}".format(action))
                if self.video_generator is not None:
                    img = env.render()
                    self.video_generator.make_image(img)
                
                next_state, reward, done, info = env.step(action)
                term_state = self.policy_phi(next_state).unsqueeze(0)
                should_terminate = torch.argmax(self.terminations.predict_idx(term_state, idx)) == 1
                steps += 1
                
                rewards.append(reward)
                
                if should_terminate:
                    reward = 1
                else:
                    reward = 0
                
                policy.observe(state,
                               action,
                               reward,
                               next_state,
                               done or should_terminate)
                
                option_rewards.append(reward)
                state = next_state
            
            if should_terminate:
                self._video_log("policy hit termination")
            if done:
                self._video_log("environment terminated")
            
            if self.video_generator is not None:
                img = env.render()
                self.video_generator.make_image(img)
            
            return state, info, steps, rewards, option_rewards, states, infos
    














