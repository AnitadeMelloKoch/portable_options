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

from portable.option.divdis.divdis_option import DivDisOption
from portable.option.memory import SetDataset
from experiments.experiment_logger import VideoGenerator
from portable.option.divdis.divdis_classifier import transform

@gin.configurable
class FactoredAdvancedMinigridDivDisExperiment():
    def __init__(self,
                 base_dir,
                 experiment_name,
                 seed,
                 policy_phi,
                 use_gpu,
                 make_videos,
                 create_plots):
        
        self.experiment_name = experiment_name
        self.seed = seed 
        self.use_gpu = use_gpu
        
        self.base_dir = os.path.join(base_dir, experiment_name, str(seed))
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.save_dir = os.path.join(self.base_dir, 'checkpoints')
        self.plot_dir = os.path.join(self.base_dir, 'plots')
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        
        set_seed(seed)
        
        if make_videos:
            self.video_generator = VideoGenerator(os.path.join(self.base_dir, "videos"))
        else:
            self.video_generator = None
        
        self.option = DivDisOption(use_gpu=use_gpu,
                                   log_dir=self.log_dir,
                                   save_dir=self.save_dir,
                                   policy_phi=policy_phi,
                                   video_generator=self.video_generator,
                                   plot_dir=None if not create_plots else self.plot_dir
                                   )
        
        log_file = os.path.join(self.log_dir, 
                                "{}.log".format(datetime.datetime.now()))
        logging.basicConfig(filename=log_file, 
                            format='%(asctime)s %(levelname)s: %(message)s',
                            level=logging.INFO)
        logging.info("[experiment] Beginning experiment {} seed {}".format(self.experiment_name, self.seed))
        
        
    def save(self):
        self.option.save()
    
    def load(self):
        self.option.load()
    
    def add_datafiles(self,
                      positive_files,
                      negative_files,
                      unlabelled_files):
        self.option.add_datafiles(positive_files,
                                  negative_files,
                                  unlabelled_files)
    
    def train_termination(self, 
                          epochs,
                          test_positive_files,
                          test_negative_files):
        for epoch in range(epochs):
            self.option.terminations.train(1, epoch)
            self.test_terminations(test_positive_files,
                                   test_negative_files)
    
    def run_rollout(self,
                    env,
                    obs,
                    info,
                    idx,
                    seed,
                    eval,
                    perfect_term=lambda x: False):
        # run one environment rollout for a specific option
        if eval is True:
            if self.video_generator is not None:
                self.video_generator.episode_start()
            
            _, _, steps, true_rewards, option_rewards, _, _ = self.option.eval_policy(idx,
                                                                           env,
                                                                           obs,
                                                                           info,
                                                                           seed)
            
            if self.video_generator is not None:
                self.video_generator.episode_end("seed{}_idx{}".format(seed, idx))

            return steps, true_rewards, option_rewards
        
        else:
            _, _, steps, _, option_rewards, _, _ = self.option.train_policy(idx,
                                                                            env,
                                                                            obs,
                                                                            info,
                                                                            seed,
                                                                            perfect_term)
        
            return steps, option_rewards

    def test_terminations(self,
                          test_positive_files,
                          test_negative_files):
        dataset_positive = SetDataset(max_size=1e6,
                                      batchsize=64)
        
        dataset_negative = SetDataset(max_size=1e6,
                                      batchsize=64)
        
        dataset_positive.set_transform_function(transform)
        dataset_negative.set_transform_function(transform)
        
        dataset_positive.add_true_files(test_positive_files)
        dataset_negative.add_false_files(test_negative_files)
        
        counter = 0
        accuracy = np.zeros(self.option.terminations.head_num)
        accuracy_pos = np.zeros(self.option.terminations.head_num)
        accuracy_neg = np.zeros(self.option.terminations.head_num)
        
        for _ in range(dataset_positive.num_batches):
            counter += 1
            x, y = dataset_positive.get_batch()
            pred_y = self.option.terminations.predict(x)
            pred_y = pred_y.cpu()
            
            for idx in range(self.option.num_heads):
                pred_class = torch.argmax(pred_y[:,idx,:], dim=1).detach()
                accuracy_pos[idx] += (torch.sum(pred_class==y).item())/len(y)
                accuracy[idx] += (torch.sum(pred_class==y).item())/len(y)
        
        accuracy_pos /= counter
        
        total_count = counter
        counter = 0
        
        for _ in range(dataset_negative.num_batches):
            counter += 1
            x, y = dataset_negative.get_batch()
            pred_y = self.option.terminations.predict(x)
            pred_y = pred_y.cpu()
            
            for idx in range(self.option.num_heads):
                pred_class = torch.argmax(pred_y[:,idx,:], dim=1).detach()
                accuracy_neg[idx] += (torch.sum(pred_class==y).item())/len(y)
                accuracy[idx] += (torch.sum(pred_class==y).item())/len(y)
        
        accuracy_neg /= counter
        total_count += counter
        
        accuracy /= total_count
        
        weighted_acc = (accuracy_pos + accuracy_neg)/2
        
        logging.info("============= Classifiers evaluated =============")
        for idx in range(self.option.num_heads):
            logging.info("idx:{:.4f} true accuracy: {:.4f} false accuracy: {:.4f} total accuracy: {:.4f} weighted accuracy: {:.4f}".format(
                idx,
                accuracy_pos[idx],
                accuracy_neg[idx],
                accuracy[idx],
                weighted_acc[idx])
            )
        logging.info("=================================================")
        
        return accuracy, weighted_acc
    
    def perfect_term_policy_train(self,
                                  env,
                                  idx,
                                  seed):
        obs, info = env.reset()
        
        _, _, steps, _, option_rewards, _, _ = self.option.env_train_policy(idx,
                                                                            env,
                                                                            obs,
                                                                            info,
                                                                            seed)
        
        return steps, option_rewards
    
    def train_policy(self,
                     max_steps,
                     min_performance,
                     envs,
                     idx,
                     seed,
                     perfect_term=lambda x: False):
        
        total_steps = 0
        option_rewards = deque(maxlen=200)
        episode = 0
        
        while total_steps < max_steps:
            env = random.choice(envs)
            
            rand_num = np.random.randint(low=0, high=20)
            obs, info = env.reset(agent_reposition_attempts=rand_num)
            
            steps, rewards = self.run_rollout(env=env,
                                              obs=obs,
                                              info=info,
                                              idx=idx,
                                              seed=seed,
                                              eval=False,
                                              perfect_term=perfect_term)
            total_steps += steps
            option_rewards.append(sum(rewards))
            
            if episode % 400 == 0:
                logging.info("idx {} steps: {} average reward: {}".format(idx,
                                                                        total_steps,
                                                                        np.mean(option_rewards)))

            episode += 1
            
            if total_steps > 800000 and np.mean(option_rewards) > min_performance:
                logging.info("idx {} reached required performance with average reward: {} at step {}".format(idx,
                                                                                                             np.mean(option_rewards),
                                                                                                             total_steps))
                return
        
        logging.info("idx {} steps: {} average reward: {}".format(idx,
                                                                  total_steps,
                                                                  np.mean(option_rewards)))

    
    def evaluate_policy(self,
                        envs,
                        trials_per_env,
                        idx,
                        seed):
        # env needs true termination function as reward function
        
        true_rewards = []
        option_rewards = []
        
        for env in envs:
            for _ in range(trials_per_env):
                rand_num = np.random.randint(low=0, high=20)
                
                obs, info = env.reset(agent_reposition_attempts=rand_num)
            
                steps, true_reward, option_reward = self.run_rollout(env,
                                                                     obs=obs,
                                                                     info=info,
                                                                     idx=idx,
                                                                     seed=seed,
                                                                     eval=True)
                
                true_rewards.append(np.sum(true_reward))
                option_rewards.append(np.sum(option_reward))
        
        logging.info("="*20)
        logging.info("="*20)
        logging.info("[eval seed-{} idx{}] steps: {} option rewards: {} true rewards: {}".format(seed,
                                                                                                 idx,
                                                                                                 steps, 
                                                                                                 np.mean(option_rewards),
                                                                                                 np.mean(true_rewards)))
        logging.info("="*20)
        logging.info("="*20)
    