from typing import Any
from portable.option.markov import MarkovOption
from portable.option.policy.agents import evaluating
from contextlib import nullcontext
import os
import pickle
import numpy as np
from collections import deque
from portable.option.sets.models.nn_classifier import NNClassifier
import gin
import torch
from portable.option.policy.agents import EnsembleAgent
import matplotlib.pyplot as plt

@gin.configurable
class NNMarkovOption(MarkovOption):
    """
    Markov option that uses images to determine the 
    initiation and termination sets using a neural network
    """
    def __init__(self,
                 initiation_states,
                 initiation_labels,
                 termination,
                 initial_policy: EnsembleAgent,
                 max_option_steps,
                 initiation_votes,
                 termination_votes,
                 min_required_interactions,
                 success_rate_required,
                 assimilation_min_required_interactions,
                 assimilation_success_rate_required,
                 classifier_type,
                 image_height,
                 image_width,
                 num_channels,
                 classifier_train_epochs,
                 use_gpu,
                 lr,
                 save_file,
                 use_log=True):
        super().__init__(use_log)
        
        assert classifier_type in ["cnn", "mlp"]
        
        self.save_file = save_file
        self.initiation = NNClassifier(classifier_type,
                                       image_height,
                                       image_width,
                                       num_channels,
                                       use_gpu,
                                       lr)
        self.initiation.add_data(initiation_states,
                                 initiation_labels)
        self.classifier_train_epochs = classifier_train_epochs
        self.initiation.train(self.classifier_train_epochs)
        self.initiation_votes = initiation_votes
        
        self.policy = initial_policy.initialize_new_policy()
        
        self.termination = termination
        self.termination_votes = termination_votes
        
        self.option_timeout = max_option_steps
        
        self.performance = deque(maxlen=min_required_interactions)
        self.min_interactions = min_required_interactions
        self.success_rate = success_rate_required
        self.assimilation_performance = deque(maxlen=min_required_interactions)
        self.assimilation_min_interactions = assimilation_min_required_interactions
        self.assimilation_success_rate_required = assimilation_success_rate_required
        self.policy.store_buffer(save_file=self.save_file)
    
    @staticmethod
    def _get_save_paths(path):
        policy = os.path.join(path, 'policy')
        initiation = os.path.join(path, 'initiation')
        termination = os.path.join(path, 'termination')
        
        return policy, initiation, termination
    
    def add_negative_data(self, data):
        self.initiation.add_data(data, [0]*len(data))
    
    def save(self, path: str):
        policy_path, initiation_path, termination_path = self._get_save_paths(path)
        
        os.makedirs(policy_path, exist_ok=True)
        os.makedirs(initiation_path, exist_ok=True)
        os.makedirs(termination_path, exist_ok=True)
        
        self.policy.save(policy_path)
        self.initiation.save(initiation_path)
        with open(os.path.join(termination_path, 'termination.pkl'), "wb") as f:
            pickle.dump(self.termination, f)
        
    def load(self, path:str):
        policy_path, initiation_path, termination_path = self._get_save_paths(path)
        
        self.policy.load(policy_path)
        self.initiation.load(initiation_path)
        with open(os.path.join(termination_path, 'termination.pkl'), "rb") as f:
            self.termination = pickle.load(f)
    
    def can_initiate(self, 
                     state,
                     info):
        prediction = self.initiation.predict(state)
        return torch.argmax(prediction) == 1
    
    def can_terminate(self, state):
        return np.array_equal(state, self.termination)
    
    def run(self,
            env,
            state,
            info,
            evaluate):
        
        steps = 0
        rewards = []
        states = []
        
        self.policy.load_buffer(save_file=self.save_file)
        self.policy.move_to_gpu()
        
        with evaluating(self.policy) if evaluate else nullcontext():
            while steps < self.option_timeout:
                steps += 1
                states.append(state)
                
                action = self.policy.act(state)
                
                # print("PRIMITIVE ACTION: {}".format(action))
                
                next_state, reward, done, info = env.step(action)
                
                rewards.append(reward)
                
                should_terminate = self.can_terminate(next_state)
                
                # fig = plt.figure(num=1, clear=True)
                # ax = fig.add_subplot()
                # ax.imshow(np.transpose(state, axes=[1,2,0]))
                # plt.show(block=False)
                # print("terminate: {}".format(should_terminate))
                # print("done: {}".format(done))
                # input("Primitive action. Continue?")
                
                # overwrite reward with reward for option
                if should_terminate:
                    reward = 1
                else:
                    reward = 0
                
                self.policy.observe(state, action, reward, next_state, done or should_terminate)
                
                if done or should_terminate:
                    if should_terminate:
                        self.log('[markov option] option chose to terminate')
                        if not evaluate:
                            self._option_success({"states": states})
                        self.policy.store_buffer(save_file=self.save_file)
                        self.policy.move_to_cpu()
                        info["option_timed_out"] = False
                        return next_state, rewards, done, info, steps
                    
                    if done:
                        if not evaluate:
                            self._option_fail({"states": states})
                        self.policy.store_buffer(save_file=self.save_file)
                        self.policy.move_to_cpu()
                        info["option_timed_out"] = False
                        return next_state, rewards, done, info, steps
                
                state = next_state

        self.log('[markov option] execution timed out')
        
        if not evaluate:
            self._option_fail({"states": states})
        
        self.policy.store_buffer(save_file=self.save_file)
        self.policy.move_to_cpu()
        
        info["option_timed_out"] = True
        
        return next_state, rewards, done, info, steps
    
    def can_assimilate(self):
        if len(self.assimilation_performance) < self.assimilation_min_interactions:
            return None
        if np.mean(self.assimilation_performance) >= self.assimilation_success_rate_required:
            return True
        else:
            return False
    
    def is_well_trained(self):
        if len(self.performance) < self.min_interactions:
            return False
        if np.mean(self.performance) >= self.assimilation_success_rate_required:
            return True
        else:
            return False
    
    def assimilate_run(self,
                       env,
                       state,
                       info):
        steps = 0
        rewards = []
        
        self.policy.load_buffer()
        self.policy.move_to_gpu()
        
        with evaluating(self.policy):
            while steps < self.option_timeout:
                steps += 1
                action = self.policy.act(state)
                
                next_state, reward, done, info = env.step(action)
                
                rewards.append(reward)
                
                should_terminate = self.can_terminate(state)
                
                reward = 1 if should_terminate else 0
                
                self.policy.observe(state, action, reward, next_state, done or should_terminate)
                
                if done or should_terminate:
                    if should_terminate:
                        self.log('[assimilate test] Option chose to terminate')
                        self.assimilation_performance.append(1)
                        
                        return next_state, rewards, done, info, steps
                    
                    if done:
                        self.log('[assimilate test] Episode ended but option did not conclude')
                        self.assimilation_performance.append(0)
                        return next_state, rewards, done, info, steps
                
                state = next_state
            self.log('[assimilation test] option timed out')
            
            self.assimilation_performance.append(0)
            
            self.policy.store_buffer()
            self.policy.move_to_cpu()
            
            return next_state, rewards, done, info, steps
    
    def _option_success(self, success_data: dict):
        
        states = success_data["states"]
        self.initiation.add_data(states, [1]*len(states))
        self.initiation.train(self.classifier_train_epochs//10)
        
        self.performance.append(1)
        self.log('[Markov option success] Success Rate: {} Num interactions: {}'.format(np.mean(self.performance), len(self.performance)))
        
    def _option_fail(self, failure_data: dict):
        states = failure_data["states"]
        
        self.initiation.add_data(states, [0]*len(states))
        self.initiation.train(self.classifier_train_epochs//10)
        
        self.performance.append(0)
        self.log('[Markov option fail] Success Rate: {} Num interactions: {}'.format(np.mean(self.performance), len(self.performance)))
        
    






