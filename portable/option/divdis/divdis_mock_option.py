import logging 
import os 
import numpy as np 
import gin 
import random 
import torch 
import pickle

from portable.option.divdis.divdis_classifier import DivDisClassifier
from portable.option.divdis.policy.policy_and_initiation import PolicyWithInitiation
from portable.option.policy.agents import evaluating
import matplotlib.pyplot as plt 
from collections import deque

@gin.configurable 
class DivDisMockOption():
    def __init__(self,
                 use_gpu,
                 log_dir,
                 save_dir,
                 terminations,
                 
                 policy_phi,
                 video_generator=None,
                 plot_dir=None):
        
        self.use_gpu = use_gpu
        self.save_dir = save_dir
        self.policy_phi = policy_phi
        self.log_dir = log_dir
        
        self.terminations = terminations
        
        self.num_heads = len(terminations)
        
        self.policies = [
            {} for _ in range(self.num_heads)
        ]
        
        self.initiable_policies = None
        self.video_generator = video_generator
        self.make_plots = False
        
        if plot_dir is not None:
            self.make_plots = True
            self.plot_dir = plot_dir
            self.term_states = []
            self.missed_term_states = []
    
    def _video_log(self, line):
        if self.video_generator is not None:
            self.video_generator.add_line(line)
    
    def _get_termination_save_path(self):
        return os.path.join(self.save_dir, 'termination')
    
    def save(self):
        os.makedirs(self.save_dir)
        for idx, policies in enumerate(self.policies):
            for key in policies.keys():
                policies[key].save(os.path.join(self.save_dir, "{}_{}".format(idx, key)))
        
            with open(os.path.join(self.save_dir, "{}_policy_keys.pkl".format(idx)), "wb") as f:
                pickle.dump(list(policies.keys()), f)
    
    def load(self):
        for idx, policies in enumerate(self.policies):
            with open(os.path.join(self.save_dir, "{}_policy_keys.pkl".format(idx)), "rb") as f:
                keys = pickle.load(f)
            for key in keys:
                policies[key] = PolicyWithInitiation(use_gpu=self.use_gpu,
                                                     policy_phi=self.policy_phi)
                policies[key].load(os.path.join(self.save_dir, "{}_{}".format(idx, key)))
            
        
        return
        if os.path.exists(self._get_termination_save_path()):
            # print in green text
            print("\033[92m {}\033[00m" .format("Termination model loaded"))
            self.terminations.load(self._get_termination_save_path())
        else:
            # print in red text
            print("\033[91m {}\033[00m" .format("No Checkpoint found. No model has been loaded"))
    
    def add_policy(self, 
                   term_idx):
        self.policies[term_idx].append(PolicyWithInitiation(use_gpu=self.use_gpu,
                                                            policy_phi=self.policy_phi))
    
    # def find_possible_policy(self, obs):
    #     policy_idxs = []
        
    #     for policies in self.policies:
    #         idxs = []
    #         for idx in range(len(policies)):
    #             if policies[idx].can_initiate(obs):
    #                 idxs.append(idx)
    #         policy_idxs.append(idxs)
        
    #     self.initiable_policies = policy_idxs
        
    #     return policy_idxs
    
    def find_possible_policies(self, seed):
        mask = [False]*self.num_heads
        for idx, policies in enumerate(self.policies):
            if seed in policies.keys():
                mask[idx] = True
        
        return mask
    
    def add_datafiles(self,
                      positive_files,
                      negative_files,
                      unlabelled_files):
        return
    
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
        
        while not (done or should_terminate):
            states.append(state)
            infos.append(info)
            
            action = policy.act(state)
            
            next_state, reward, done, info = env.step(action)
            should_terminate = self.terminations[idx](state,
                                                      env)
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
        
        # if should_terminate:
        #     # should save positive examples and create classifier if it hasn't been
        #     pass
        # if done:
        #     pass
        
        return state, info, steps, rewards, option_rewards, states, infos
    
    def bootstrap_policy(self,
                         idx,
                         envs,
                         max_steps,
                         min_performance,
                         seed):
        total_steps = 0
        option_rewards = deque(maxlen=200)
        episode = 0
        
        while total_steps < max_steps:
            env = random.choice(envs)
            rand_num = np.random.randint(low=0, high=50)
            obs, info = env.reset(agent_reposition_attempts=rand_num)
            
            _, _, steps, _, rewards, _, _ = self.train_policy(idx,
                                                              env,
                                                              obs,
                                                              info,
                                                              seed)
            total_steps += steps
            option_rewards.append(sum(rewards))
            
            if episode % 400 == 0:
                logging.info("idx {} steps: {} average reward: {}".format(idx,
                                                                          total_steps,
                                                                          np.mean(option_rewards)))
            episode += 1
            
            if total_steps > 200000 and np.mean(option_rewards) > min_performance:
                logging.info("idx {} reached required performance with average reward: {} at step {}".format(idx,
                                                                                                             np.mean(option_rewards),
                                                                                                             total_steps))
                break
        
        self.save()
            
            
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
        policy.move_to_gpu()
        
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
                should_terminate = self.terminations[idx](state,
                                                          env)
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
    














