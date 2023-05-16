import torch 
import random 
import numpy as np 
from experiments.minigrid.doorkey.core.agents.rainbow import Rainbow
from portable.option.minigrid_option import MinigridOption
import os
from portable.option.sets.utils import VOTE_FUNCTION_NAMES
import logging 
import datetime 
import pandas as pd
import lzma 
import dill 
import gin

@gin.configurable
class MinigridOptionExperiment():
    def __init__(self,
                 base_dir,
                 experiment_name,
                 training_seed,
                 training_envs,
                 random_seed,
                 create_env_function,
                 num_levels,
                 create_agent_function,
                 action_space,
                 policy_phi,
                 num_frames_train=10**5,
                 lr=1e-4,
                 initiation_vote_function="weighted_vote_low",
                 termination_vote_function="weighted_vote_low",
                 device_type="cpu",
                 sigma=0.5,
                 train_options=True,
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
                 train_policy_success_rate=0.8,
                 max_option_tries=5):
        
        assert device_type in ["cpu", "cuda"]
        assert initiation_vote_function in VOTE_FUNCTION_NAMES
        assert termination_vote_function in VOTE_FUNCTION_NAMES
        
        self.device = torch.device(device_type)
        
        self.make_env = create_env_function
        
        random.seed(random_seed)
        self.random_seed = random_seed
        self.name = experiment_name
        self.base_dir = os.path.join(base_dir, self.name, str(self.random_seed))
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.plot_dir = os.path.join(self.base_dir, 'plots')
        self.save_dir = os.path.join(self.base_dir, 'checkpoints')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        
        self.agent = create_agent_function(action_space,
                                        #    save_dir=os.path.join(self.save_dir, "option_agent"),
                                           gpu=0,
                                           n_input_channels=3,
                                           lr=lr,
                                           sigma=sigma)
        self.num_base_actions = 7
        self.num_options = action_space - self.num_base_actions
        self.options = [MinigridOption(self.device,
                                       initiation_vote_function,
                                       termination_vote_function,
                                       policy_phi) for _ in range(self.num_options)]
        
        for idx in range(len(initiation_positive_files)):
            self.options[idx].initiation.add_data_from_files(initiation_positive_files[idx],
                                            initiation_negative_files[idx],
                                            initiation_priority_negative_files[idx])
            self.options[idx].termination.add_data_from_files(termination_positive_files[idx],
                                            termination_negative_files[idx],
                                            termination_priority_negative_files[idx])
        
        if train_options:
            self.train_policy(training_envs,
                              train_policy_max_steps,
                              train_policy_success_rate)
            self.train_initiation(train_initiation_embedding_epochs,
                                   train_initiation_classifier_epochs)
            self.train_termination(train_termination_embedding_epochs,
                                    train_termination_classifier_epochs)
                
        self.training_env_seed = training_seed
        self.test_env_seeds = random.choices(np.arange(1000), k=num_levels)
        self.test_env_seeds = [int(x) for x in self.test_env_seeds]
        self.num_levels = num_levels
        self.num_frames_train = num_frames_train
        
        log_file = os.path.join(self.log_dir, "{}.log".format(datetime.datetime.now()))
        logging.basicConfig(filename=log_file, 
                            format='%(asctime)s %(levelname)s: %(message)s')
        logging.info("[experiment] Beginning experiment {} seed {}".format(self.name, self.random_seed))
        logging.info("======== HYPERPARAMETERS ========")
        logging.info("Train options: {}".format(train_options))
        logging.info("Initiation vote function: {}".format(initiation_vote_function))
        logging.info("Termination vote function: {}".format(termination_vote_function))
        logging.info("Initiation embedding epochs: {}".format(train_initiation_embedding_epochs))
        logging.info("Initiation classifier epochs: {}".format(train_initiation_classifier_epochs))
        logging.info("Termination embedding epochs: {}".format(train_termination_embedding_epochs))
        logging.info("Termination classifier epochs: {}".format(train_termination_classifier_epochs))
        logging.info("Train policy max steps: {}".format(train_policy_max_steps))
        logging.info("Train policy success rate: {}".format(train_poilcy_success_rate))
        logging.info("Max option tries: {}".format(max_option_tries))
        logging.info("Train seed: {}".format(training_seed))
        logging.info("Number of levels in experiment: {}".format(num_levels))
        logging.info("Test seeds: {}".format(self.test_env_seeds))
        
        self.trial_data = pd.DataFrame([],
                                       columns=[
                                           'reward',
                                           'seed',
                                           'frames',
                                           'env_num'
                                       ])
    
    def save(self):
        self.agent.save()
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
    
    def train_policy(self, training_envs, max_steps, success_rate):
        for idx in range(len(training_envs)):
            self.options[idx].bootstrap_policy(training_envs[idx],
                                               max_steps,
                                               success_rate)
    
    def train_initiation(self, embedding_epochs, classifier_epochs):
        for idx in range(len(self.options)):
            self.options[idx].initiation.train(embedding_epochs,
                                               classifier_epochs)
    
    def train_termination(self, embedding_epochs, classifier_epochs):
        for idx in range(len(self.options)):
            self.options[idx].termination.train(embedding_epochs,
                                               classifier_epochs)
    
    def train_test_envs(self, 
                        episodes_per_level):
        for idx, env_seed in enumerate(self.test_env_seeds):
            env = self.make_env(env_seed)
            total_steps = 0
            ep = 0
            while steps < self.num_frames_train:
                episode_rewards, steps = self.run_episode(env)
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
                
    def run_episode(self, env):
        obs, info = env.reset()
        done = False
        episode_reward = 0.
        rewards = []
        trajectory = []
        steps = 0
        
        while not done:
            action = self.agent.act(obs)
            next_obs, reward, done, info, step_num = self.run_one_step(obs, info, env)
            rewards.append(reward)
            trajectory.append((obs, action, reward, next_obs, done, info["needs_reset"]))
            
            obs = next_obs
            episode_reward += reward
            steps += step_num

        self.agent.experience_replay(trajectory)
        return rewards, steps
    
    def run_one_step(self, state, info, env):
        action = self.agent.act(state)
        if action < self.num_base_actions:
            next_obs, reward,done, info = env.step(action)
            return next_obs, reward, done, info, 1
        
        option_idx = action - self.num_base_actions
        if not self.options[option_idx].can_initiate(state, 
                                                 (info["player_x"], info["player_y"], info["has_key"], info["door_open"])):
            next_obs, reward,done, info = env.step(5)
            return next_obs, reward, done, info, 1
        return self.options[option_idx].run(env,
                                     state,
                                     info,
                                     eval=False)