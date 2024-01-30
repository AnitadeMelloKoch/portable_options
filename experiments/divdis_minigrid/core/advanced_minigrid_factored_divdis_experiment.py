import logging 
import datetime 
import os 
import gin 
import numpy as np 
from portable.utils.utils import set_seed 
from torch.utils.tensorboard import SummaryWriter 
import torch 

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
                 make_videos):
        
        self.experiment_name = experiment_name
        self.seed = seed 
        self.use_gpu = use_gpu
        
        self.base_dir = os.path.join(base_dir, experiment_name, str(seed))
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.save_dir = os.path.join(self.base_dir, 'checkpoints')
        
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
                                   video_generator=self.video_generator)
        
        
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
    
    def train_termination(self, epochs):
        self.option.terminations.train(epochs)
    
    def run_rollout(self,
                    env,
                    idx,
                    seed,
                    eval):
        # run one environment rollout for a specific option
        obs, info = env.reset()
        
        if eval:
            if self.video_generator is not None:
                self.video_generator.episode_start()
            
            _, _, steps, _, option_rewards, _, _ = self.option.eval_policy(idx,
                                                                           env,
                                                                           obs,
                                                                           info,
                                                                           seed)
            
            if self.video_generator is not None:
                self.video_generator.episode_end("seed{}_idx{}".format(seed, idx))
        
        else:
            _, _, steps, _, option_rewards, _, _ = self.option.train_policy(idx,
                                                                            env,
                                                                            obs,
                                                                            info,
                                                                            seed)
        
        return steps, option_rewards

    def test_terminations(self,
                          test_positive_files,
                          test_negative_files):
        dataset = SetDataset(max_size=1e6,
                             batchsize=64)
        
        dataset.set_transform_function(transform)
        
        dataset.add_true_files(test_positive_files)
        dataset.add_false_files(test_negative_files)
        
        counter = 0
        accuracy = np.zeros(self.option.terminations.head_num)
        
        for _ in range(dataset.num_batches):
            counter += 1
            x, y = dataset.get_batch()
            pred_y = self.option.terminations.predict(x)
            pred_y = pred_y.cpu()
            
            for idx in range(self.option.num_heads):
                pred_class = torch.argmax(pred_y[:,idx,:], dim=1).detach()
                accuracy[idx] += (torch.sum(pred_class==y).item())/len(y)
                
        return accuracy/counter
    
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