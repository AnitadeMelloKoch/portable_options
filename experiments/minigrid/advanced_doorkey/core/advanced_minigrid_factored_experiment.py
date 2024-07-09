import logging
import datetime
import os 
import random 
import gin 
import torch 
import lzma 
import pickle 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from portable.utils.utils import set_seed
from torch.utils.tensorboard import SummaryWriter
from collections import deque

from portable.option import AttentionOption
from portable.option.ensemble.custom_attention import MockAutoEncoder
from portable.option.memory import SetDataset

from portable.agent.option_agent import OptionAgent

from experiments.experiment_logger import VideoGenerator

@gin.configurable
class AdvancedMinigridFactoredExperiment():
    def __init__(self,
                 base_dir,
                 experiment_name,
                 training_seed,
                 experiment_seed,
                 markov_option_builder,
                 policy_phi,
                 num_instances_per_option=10,
                 num_primitive_actions=7,
                 policy_lr=1e-4,
                 policy_max_steps=1e6,
                 policy_success_threshold=0.98,
                 use_gpu=False,
                 names=None,
                 termination_oracles=None,
                 make_videos=False):

        # for now this is only testing policy when given a factored state 
        # representation as the "embedding"
        
        self.training_seed=training_seed
        self.policy_lr=policy_lr
        self.policy_max_steps=policy_max_steps
        self.policy_success_threshold=policy_success_threshold
        self.use_gpu=use_gpu
        self.num_primitive_actions=num_primitive_actions
        self.num_intances_per_option=num_instances_per_option
        
        # have to use oracle for now
        self.use_oracle_for_term=True
        assert termination_oracles is not None
        self.termination_oracles = termination_oracles
        
        self.names=names
        self.embedding = MockAutoEncoder()
        self.policy_phi=policy_phi
        self.markov_option_builder=markov_option_builder
        
        set_seed(experiment_seed)
        self.seed = experiment_seed
        self.name = experiment_name
        self.base_dir = os.path.join(base_dir, experiment_name, str(experiment_seed))
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.plot_dir = os.path.join(self.base_dir, 'plots')
        self.save_dir = os.path.join(self.base_dir, 'checkpoints')
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=self.log_dir)
        log_file = os.path.join(self.log_dir, 
                                "{}.log".format(datetime.datetime.now()))
        
        if make_videos:
            self.video_generator = VideoGenerator(os.path.join(self.base_dir, "videos"))
        else:
            self.video_generator = None
        
        logging.basicConfig(filename=log_file, 
                            format='%(asctime)s %(levelname)s: %(message)s',
                            level=logging.INFO)
        logging.info("[experiment] Beginning experiment {} seed {}".format(self.name, self.seed))
        logging.info("======== HYPERPARAMETERS ========")
        logging.info("Experiment seed: {}".format(experiment_seed))
        logging.info("Training seed: {}".format(training_seed))
        
        self.option = AttentionOption(use_gpu=use_gpu,
                                      log_dir=os.path.join(self.log_dir, 'option'),
                                      markov_option_builder=markov_option_builder,
                                      embedding=self.embedding,
                                      policy_phi=policy_phi,
                                      num_actions=num_primitive_actions,
                                      use_oracle_for_term=self.use_oracle_for_term,
                                      termination_oracle=termination_oracles,
                                      save_dir=self.save_dir,
                                      video_generator=self.video_generator,
                                      option_name=names,
                                      factored_obs=True)

        self.rewards = [{}]*self.option.policy.num_modules
        
        self.steps = [{}]*self.option.policy.num_modules
    
    def _video_log(self, line):
        if self.video_generator is not None:
            self.video_generator.add_line(line)
    
    def reset_option(self):
        self.option = AttentionOption(use_gpu=self.use_gpu,
                                      log_dir=os.path.join(self.log_dir, 'option'),
                                      markov_option_builder=self.markov_option_builder,
                                      embedding=self.embedding,
                                      policy_phi=self.policy_phi,
                                      num_actions=self.num_primitive_actions,
                                      use_oracle_for_term=self.use_oracle_for_term,
                                      termination_oracle=self.termination_oracles,
                                      save_dir=self.save_dir,
                                      video_generator=self.video_generator,
                                      option_name=self.names,
                                      factored_obs=True)
    
    def save(self):
        self.option.save()
    
    def load(self):
        self.option.load()
    
    def train_policy(self,
                     training_envs):
        self.option.bootstrap_policy(training_envs,
                                     self.policy_max_steps,
                                     self.policy_success_threshold)
        
        self.option.save()
    
    def add_datafiles(self,
                      positive_files,
                      negative_files):
        
        self.option.initiation.add_data_from_files(positive_files,
                                                   negative_files)
    
    def train_classifier(self, epochs):
        self.option.initiation.train(epochs)
        print(self.option.initiation.classifier.get_attention_masks())
    
    def test_classifier(self,
                        test_positive_files,
                        test_negative_files):
        
        dataset = SetDataset(max_size=1e6,
                             batchsize=64)
        
        dataset.add_true_files(test_positive_files)
        dataset.add_false_files(test_negative_files)
        
        counter = 0
        accuracy = []
        loss = []
        
        for _ in range(dataset.num_batches):
            counter += 1
            x, y = dataset.get_batch()
            l, acc = self.option.initiation.batch_pred(x,y)
            
            loss.append(l)
            accuracy.append(acc)
        
        loss = np.array(loss)
        loss = np.sum(loss)/counter
        accuracy = np.array(accuracy)
        accuracy = np.sum(accuracy)/counter
        
        return loss, accuracy
    
    def run_episode(self,
                    env,
                    policy_idx,
                    video_name,
                    num_train_envs):
        
        self.video_generator.episode_start()
        
        obs, info = env.reset()
        done = False
        episode_reward = 0
        rewards = []
        steps = 0
        
        if self.video_generator is not None:
            self.video_generator.episode_start()
        
        _, rewards, _, _, steps = self.option.run(env, 
                        obs, 
                        info,
                        eval=True,
                        false_states=[],
                        policy_leader=policy_idx)
        
        if num_train_envs not in self.rewards[policy_idx]:
            self.rewards[policy_idx][num_train_envs] = []
        
        self.rewards[policy_idx][num_train_envs].append(sum(rewards))
        
        if num_train_envs not in self.steps[policy_idx]:
            self.steps[policy_idx][num_train_envs] = []
        
        self.steps[policy_idx][num_train_envs].append(steps)
        
        with open(os.path.join(self.save_dir, 'rewards.pkl'), 'wb') as fp:
            pickle.dump(self.rewards, fp)
        
        with open(os.path.join(self.save_dir, 'steps.pkl'), 'wb') as fp:
            pickle.dump(self.steps, fp)
        
        self.video_generator.episode_end(video_name)
            
            
            
        
