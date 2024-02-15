import logging 
import datetime 
import os 
import gin 
import numpy as np 
from portable.utils.utils import set_seed 
from torch.utils.tensorboard import SummaryWriter 
import torch 
from collections import deque
import random
import matplotlib.pyplot as plt

from portable.option.divdis.divdis_mock_option import DivDisMockOption
from experiments.experiment_logger import VideoGenerator
from portable.agent.option_agent import OptionAgent

@gin.configurable
class FactoredAdvancedMinigridDivDisMetaExperiment():
    def __init__(self,
                 base_dir,
                 experiment_name,
                 seed,
                 meta_policy_phi,
                 use_gpu,
                 action_agent,
                 option_agent,
                 terminations,
                 num_options,
                 num_primitive_actions,
                 discount_rate=0.9,
                 image_state=True,
                 make_videos=False,
                 option_policy_phi=None) -> None:
        
        self.name = experiment_name,
        self.seed = seed 
        self.use_gpu = use_gpu
        
        self.base_dir = os.path.join(base_dir, experiment_name, str(seed))
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.save_dir = os.path.join(self.base_dir, 'checkpoints')
        self.plot_dir = os.path.join(self.base_dir, 'plots')
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        
        set_seed(seed)
        
        log_file = os.path.join(self.log_dir, 
                                "{}.log".format(datetime.datetime.now()))
        logging.basicConfig(filename=log_file, 
                            format='%(asctime)s %(levelname)s: %(message)s',
                            level=logging.INFO)
        logging.info("[experiment] Beginning experiment {} seed {}".format(self.name, self.seed))
        
        if make_videos:
            self.video_generator = VideoGenerator(os.path.join(self.base_dir, "videos"))
        else:
            self.video_generator = None
        
        self.meta_agent = OptionAgent(action_agent=action_agent,
                                      option_agent=option_agent,
                                      use_gpu=use_gpu,
                                      phi=meta_policy_phi,
                                      video_generator=self.video_generator)
        
        
        if option_policy_phi is None:
            option_policy_phi = meta_policy_phi
        
        self.options = []
        
        assert len(terminations) == num_options
        self.num_options = num_options
        self.num_primitive_actions = num_primitive_actions
        
        for idx, termination_list in enumerate(terminations):
            self.options.append(DivDisMockOption(use_gpu=use_gpu,
                                                 log_dir=os.path.join(self.log_dir, "option_{}".format(idx)),
                                                 save_dir=os.path.join(self.save_dir, "option_{}".format(idx)),
                                                 terminations=termination_list,
                                                 policy_phi=option_policy_phi,
                                                 video_generator=self.video_generator,
                                                 plot_dir=os.path.join(self.plot_dir, "option_{}".format(idx))))
        
        self.num_heads = self.options[0].num_heads
        self.gamma = discount_rate
    
    def save(self):
        pass
    
    def load(self):
        pass
    
    def _video_log(self, line):
        if self.video_generator is not None:
            self.video_generator.add_line(line)
    
    def train_option_policies(self, 
                              train_envs_list, 
                              seed, 
                              max_steps):
        for o_idx, option_train_envs in enumerate(train_envs_list):
            for t_idx, term_train_envs in enumerate(option_train_envs):
                self.options[o_idx].bootstrap_policy(t_idx,
                                                     term_train_envs,
                                                     max_steps,
                                                     0.98,
                                                     seed)
    
    def get_masks_from_seed(self,
                            seed):
        
        action_mask = [False]*self.num_options
        option_masks = []
        
        for idx in range(self.num_primitive_actions):
            action_mask[idx] = True
            option_masks.append([False]*self.num_heads)
        
        for idx, option in enumerate(self.options):
            option_mask = option.find_possible_policies(seed)
            option_masks.append(option_mask)
            if any(option_mask):
                action_mask[self.num_primitive_actions+idx] = True
        
        return action_mask, option_masks
    
    def save_image(self, env):
        if self.video_generator is not None:
            img = env.render()
            self.video_generator.make_image(img)
    
    def train_meta_agent(self,
                         env,
                         seed,
                         max_steps,
                         min_performance):
        total_steps = 0
        episode_rewards = deque(maxlen=200)
        episode = 0
        done = False
        undiscounted_rewards = []
        
        while total_steps < max_steps:
            undiscounted_reward = 0
            
            if self.video_generator is not None:
                self.video_generator.episode_start()
            
            obs, info = env.reset()
            
            
            while not done:
                self.save_image(env)
                action_mask, option_masks = self.get_masks_from_seed(seed)
                
                action, option = self.meta_agent.act(obs,
                                                     action_mask,
                                                     option_masks)
                
                self._video_log("action: {} option: {}".format(action, option))
                
                if action < self.num_primitive_actions:
                    next_obs, reward, done, info = env.step(action)
                    undiscounted_reward += reward
                    rewards = [reward]
                    total_steps += 1
                else:
                    next_obs, info, steps, rewards, _, _, _ = self.options[action-self.num_primitive_actions].eval_policy(option,
                                                                                                                        env,
                                                                                                                        obs,
                                                                                                                        info,
                                                                                                                        seed)
                    undiscounted_reward += np.sum(rewards)
                    total_steps += steps
                
                self.meta_agent.observe(obs,
                                        action,
                                        option,
                                        rewards,
                                        next_obs,
                                        done)
            
            logging.info("Episode {} total steps: {} undiscounted reward: {}".format(episode,
                                                                                     total_steps,
                                                                                     undiscounted_reward))
            
            if (undiscounted_reward > 0 or episode%100==0) and self.video_generator is not None:
                self.video_generator.episode_end("episode_{}".format(episode))
            
            undiscounted_rewards.append(undiscounted_reward)
            episode += 1
            episode_rewards.append(undiscounted_reward)
            
            self.plot_learning_curve(episode_rewards)
            
            if total_steps > 1e6 and np.mean(episode_rewards) > min_performance:
                logging.info("Meta agent reached min performance {} in {} steps".format(np.mean(episode_rewards),
                                                                                        total_steps))
                return
    
    def plot_learning_curve(self,
                            rewards):
        x = np.arange(len(rewards))
        fig, ax = plt.subplots()
        ax.plot(x, rewards)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Sum Undiscounted Rewards")
        
        fig.savefig(os.path.join(self.plot_dir, "learning_curve.png"))
        
    
    def eval_meta_agent(self):
        pass
    
    
    
    
    













