import torch
import random
import numpy as np 
from portable.option import Option
import os 
from portable.option.sets.utils import VOTE_FUNCTION_NAMES
import logging 
import pandas as pd 
import lzma 
import dill 
import gin 
from experiments import BaseExperiment
import datetime

@gin.configurable
class FactoredMinigridSingleOptionExperiment(BaseExperiment):
        
    def __init__(self,
                 base_dir,
                 experiment_name,
                 training_env,
                 random_seed,
                 create_env_function,
                 num_levels,
                 policy_phi,
                 markov_option_builder,
                 get_latent_state_function,
                 num_frames_train=10**5,
                 initiation_vote_function="weighted_vote_low",
                 termination_vote_function="weighted_vote_low",
                 device_type="cpu",
                 train_initiation=True,
                 train_termination=True,
                 train_policy=True,
                 initiation_positive_files=[],
                 initiation_negative_files=[],
                 initiation_priority_negative_files=[],
                 train_initiation_embedding_epochs=100,
                 train_initiation_classifier_epochs=100,
                 termination_positive_files=[],
                 termination_negative_files=[],
                 termination_priority_negative_files=[],
                 train_termination_embedding_epochs=100,
                 train_termination_classifier_epochs=100,
                 train_policy_max_steps=10000,
                 train_policy_success_rate=0.8):
        
        super().__init__(base_dir, experiment_name, random_seed, create_env_function, device_type)
        
        
        assert initiation_vote_function in VOTE_FUNCTION_NAMES
        assert termination_vote_function in VOTE_FUNCTION_NAMES
        
        self.option = Option(self.device,
                             markov_option_builder=markov_option_builder,
                             get_latent_state=get_latent_state_function,
                             initiation_vote_function=initiation_vote_function,
                             termination_vote_function=termination_vote_function,
                             policy_phi=policy_phi)
        
        self.option.initiation.add_data_from_files(initiation_positive_files,
                                                   initiation_negative_files,
                                                   initiation_priority_negative_files)
        self.option.termination.add_data_from_files(termination_positive_files,
                                                    termination_negative_files,
                                                    termination_priority_negative_files)

        self.test_env_seeds = random.choices(np.arange(1000), k=num_levels)
        self.test_env_seeds = [int(x) for x in self.test_env_seeds]
        self.num_levels = num_levels
        self.num_frames_train = num_frames_train
        
        log_file = os.path.join(self.log_dir, "{}.log".format(datetime.datetime.now()))
        logging.basicConfig(filename=log_file, 
                            format='%(asctime)s %(levelname)s: %(message)s',
                            level=logging.INFO)
        logging.info("[experiment] Beginning experiment {} seed {}".format(self.name, self.random_seed))
        logging.info("======== HYPERPARAMETERS ========")
        logging.info("Train initiation: {}".format(train_initiation))
        logging.info("Train termination: {}".format(train_termination))
        logging.info("Train policy: {}".format(train_policy))
        logging.info("Initiation vote function: {}".format(initiation_vote_function))
        logging.info("Termination vote function: {}".format(termination_vote_function))
        logging.info("Initiation embedding epochs: {}".format(train_initiation_embedding_epochs))
        logging.info("Initiation classifier epochs: {}".format(train_initiation_classifier_epochs))
        logging.info("Termination embedding epochs: {}".format(train_termination_embedding_epochs))
        logging.info("Termination classifier epochs: {}".format(train_termination_classifier_epochs))
        logging.info("Train policy max steps: {}".format(train_policy_max_steps))
        logging.info("Train policy success rate: {}".format(train_policy_success_rate))
        logging.info("Number of levels in experiment: {}".format(num_levels))
        logging.info("Test seeds: {}".format(self.test_env_seeds))
        
        self.trial_data = pd.DataFrame([],
                                       columns=[
                                           'reward',
                                           'seed',
                                           'frames',
                                           'env_num'
                                       ])
        
        if train_policy:
            self.train_policy(training_env,
                              train_policy_max_steps,
                              train_policy_success_rate)
        if train_initiation:
            self.train_initiation(train_initiation_embedding_epochs,
                                  train_initiation_classifier_epochs)
        if train_termination:
            self.train_termination(train_termination_embedding_epochs,
                                   train_termination_classifier_epochs)
    
    def save(self):
        self.option.save(self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        filename = os.path.join(self.save_dir, 'experiment_data.pkl')
        with lzma.open(filename, 'wb') as f:
            dill.dump(self.trial_data, f)
            
    def load(self):
        self.option.load(self.save_dir)
        filename = os.path.join(self.save_dir, 'experiment_data.pkl')
        if os.path.exists(filename):
            with lzma.open(filename, 'rb') as f:
                self.trial_data = dill.load(f)
    
    def train_policy(self, training_env, max_steps, success_rate):
        self.option.bootstrap_policy(training_env,
                                     max_steps,
                                     success_rate)
        self.save()
        
    def train_initiation(self, embedding_epochs, classifier_epochs):
        self.option.initiation.train(embedding_epochs,
                                     classifier_epochs)
        self.save()
    
    def train_termination(self, embedding_epochs, classifier_epochs):
        self.option.termination.train(embedding_epochs,
                                      classifier_epochs)
        self.save()
        
    def train_test_envs(self):
        for idx, env_seed in enumerate(self.test_env_seeds):
            env = self.make_env(env_seed)
            total_steps = 0
            ep = 0
            while total_steps < self.num_frames_train:
                episode_rewards, steps = self.rollout(env)
                undiscounted_return = sum(episode_rewards)
                
                d = pd.DataFrame([{
                    "reward": episode_rewards,
                    "frames": total_steps,
                    "seed": env_seed,
                    "env_num": idx
                }])
                self.trial_data.append(d)
                
                print(100 * '-')
                print(f'Episode: {ep} for env seed: {env_seed}',
                f"Steps': {total_steps}",
                f'Reward: {undiscounted_return}')
                print(100 * '-')
                
                logging.log(100 * '-')
                logging.log(f'Episode: {ep} for env seed: {env_seed}',
                f"Steps': {total_steps}",
                f'Reward: {undiscounted_return}')
                logging.log(100 * '-')
                
    def rollout(self, env):
        obs, info = env.reset()
        done = False
        episode_reward = 0.
        rewards = []
        trajectory = []
        steps = 0
        
        while not done:
            action = self.agent.act(obs)
            next_obs, reward, done, info, step_num = self.option.run(env,
                                                                     obs,
                                                                     info,
                                                                     eval=False)
            rewards.append(reward)
            trajectory.append((obs, action, reward, next_obs, done, info["needs_reset"]))
            
            obs = next_obs
            episode_reward += reward 
            steps += step_num
            
        return rewards, steps
        
        
        
        