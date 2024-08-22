import logging 
import datetime 
import os 
import gin 
import numpy as np 
from portable.utils.utils import set_seed 
from torch.utils.tensorboard import SummaryWriter 
import torch 
from collections import deque
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

from portable.option.divdis.divdis_mock_option import DivDisMockOption
from portable.option.divdis.divdis_option import DivDisOption
from experiments.experiment_logger import VideoGenerator
from portable.option.memory import SetDataset

from torch.multiprocessing import Process, Pipe, set_start_method
from experiments import train_head

from portable.utils import load_init_states
from pfrl.wrappers import atari_wrappers
from experiments.monte.environment import MonteBootstrapWrapper, MonteAgentWrapper
from experiments.divdis_monte.core.monte_terminations import *

OPTION_TYPES = ["mock", "divdis"]

def policy_phi(x):
    if type(x) == np.ndarray:
        x = torch.from_numpy(x)
    x = (x/255.0).float()
    return x

def mock_check_true(x,y,z):
    return False

def make_monte_env(seed, init_state):
    env = atari_wrappers.wrap_deepmind(
        atari_wrappers.make_atari('MontezumaRevengeNoFrameskip-v4'),
        episode_life=True,
        clip_rewards=True,
        frame_stack=False
    )
    env.seed(seed)
    
    # env = MonteAgentWrapper(env, agent_space=False, stack_observations=False)
    env = MonteBootstrapWrapper(env,
                                agent_space=False,
                                list_init_states=load_init_states(init_state),
                                check_true_termination=mock_check_true,
                                list_termination_points=[(0,0,0)]*len(init_state),
                                max_steps=int(1e4))
    
    return env

@gin.configurable
class DivDisOptionExperiment():
    def __init__(self,
                 base_dir,
                 experiment_name,
                 seed,
                 gpu_id,
                 option_type,
                 config_file,
                 gin_bindings,
                 num_processes=4,
                 terminations=[],
                 option_timeout=50,
                 classifier_epochs=100,
                 train_new_policy_for_each_room=True,
                 make_videos=False,
                 env_ram_dir="."):
        assert option_type in OPTION_TYPES
        
        self.name = experiment_name
        self.seed = seed
        self.option_type = option_type
        self.option_timeout = option_timeout
        self.learn_new_policy = train_new_policy_for_each_room
        self.num_processes = num_processes
        self.config_file =config_file
        self.gin_bindings = gin_bindings
        self.env_ram_dir = env_ram_dir
        
        self.base_dir = os.path.join(base_dir, experiment_name, str(seed))
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.save_dir = os.path.join(self.base_dir, 'checkpoints')
        self.plot_dir = os.path.join(self.base_dir, 'plots')
        self.classifier_epochs = classifier_epochs
        
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        self.run_numbers = defaultdict(int)
        
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
        
        if option_type == "mock":
            self.option = DivDisMockOption(use_gpu=gpu_id,
                                           log_dir=os.path.join(self.log_dir, "option"),
                                           save_dir=os.path.join(self.save_dir, "option"),
                                           terminations=terminations,
                                           policy_phi=policy_phi,
                                           video_generator=self.video_generator,
                                           plot_dir=os.path.join(self.plot_dir, "option"),
                                           use_seed_for_initiation=True)
        elif option_type == "divdis":
            self.option = DivDisOption(use_gpu=gpu_id,
                                        log_dir=os.path.join(self.log_dir, "option"),
                                        save_dir=os.path.join(self.save_dir, "option"),
                                        policy_phi=policy_phi,
                                        video_generator=self.video_generator,
                                        plot_dir=os.path.join(self.plot_dir, "option"),
                                        use_seed_for_initiation=True)
        
        set_start_method('spawn')
        self.experiment_data = []
    
    def change_option_save(self, name):
        self.option.reset_policies()
        self.option.plot_dir = os.path.join(self.plot_dir, name)
        self.option.log_dir = os.path.join(self.log_dir, name)
        self.option.save_dir = os.path.join(self.save_dir, name)
        
        for idx in range(self.option.num_heads):
            os.makedirs(os.path.join(self.option.plot_dir, str(idx)), exist_ok=True)
    
    def save(self):
        # self.option.save()
        
        with open(os.path.join(self.save_dir, "experiment_data.pkl"), "wb") as f:
            pickle.dump(self.experiment_data, f)
    
    def load(self):
        self.option.load()
        
        with open(os.path.join(self.save_dir, "experiment_data.pkl"), "rb") as f:
            self.experiment_data = pickle.load(f)
    
    def add_datafiles(self,
                      positive_files,
                      negative_files,
                      unlabelled_files):
        self.option.add_datafiles(positive_files,
                                  negative_files,
                                  unlabelled_files)
    
    def train_classifier(self,
                         epochs=None):
        if epochs is None:
            epochs = self.classifier_epochs
        self.option.terminations.set_class_weights()
        self.option.terminations.train(epochs)
    
    def train_option(self,
                     init_states,
                     term_points,
                     seed,
                     max_steps,
                     env_idx):
        
        head_idx = 0
        processes = []
        parent_pipes, child_pipes = [], []
        for state_file in init_states:
            state_file = os.path.join(self.env_ram_dir, state_file)
        for _ in range(self.num_processes):
            if head_idx >= self.option.num_heads:
                continue
            
            parent_con, child_con = Pipe()
            
            p = Process(target=train_head, args=(head_idx, 
                                                 child_con,
                                                 max_steps,
                                                 self.option_timeout,
                                                 make_monte_env,
                                                 self.option,
                                                 seed,
                                                 self.learn_new_policy,
                                                 env_idx,
                                                 self.log_dir,
                                                 init_states,
                                                 term_points,
                                                 self.config_file,
                                                 self.gin_bindings))
            
            # train_head(
            #     head_idx, 
            #     child_con,
            #     max_steps,
            #     self.option_timeout,
            #     make_monte_env,
            #     self.option,
            #     seed,
            #     self.learn_new_policy,
            #     env_idx,
            #     self.log_dir,
            #     init_states,
            #     term_points,
            #     self.config_file,
            #     self.gin_bindings
            # )
            
            head_idx += 1
            processes.append(p)
            parent_pipes.append(parent_con)
            child_pipes.append(child_con)
            p.start()
            
        for p, parent_con in zip(processes, parent_pipes):
            exp_data = parent_con.recv()
            self.experiment_data.extend(exp_data)
            p.join()
        
        self.save()
        
    
    def test_classifiers(self,
                         test_positive_files,
                         test_negative_files):
        
        self.accuracy_pos = []
        self.accuracy_neg = []
        self.weighted_accuracy = []
        self.accuracy = []
        
        dataset_positive = SetDataset(max_size=1e6,
                                      batchsize=64)
        dataset_negative = SetDataset(max_size=1e6,
                                      batchsize=64)
        
        dataset_positive.add_true_files(test_positive_files)
        dataset_negative.add_false_files(test_negative_files)
        
        counter = 0
        accuracy = np.zeros(self.option.num_heads)
        accuracy_pos = np.zeros(self.option.num_heads)
        accuracy_neg = np.zeros(self.option.num_heads)
        
        for _ in range(dataset_positive.num_batches):
            counter += 1
            x, y = dataset_positive.get_batch()
            pred_y, _ = self.option.terminations.predict(x)
            pred_y = pred_y.cpu()
            
            for idx in range(self.option.num_heads):
                pred_class = torch.argmax(pred_y[:,idx,:], dim=1).detach()
                accuracy_pos[idx] += (torch.sum(pred_class==y).item())/len(y)
                accuracy[idx] += (torch.sum(pred_class==y).item())/len(y)
        
        accuracy_pos /= counter
        
        total_counter = counter
        counter = 0
        
        for _ in range(dataset_negative.num_batches):
            counter += 1
            x, y = dataset_negative.get_batch()
            pred_y, _ = self.option.terminations.predict(x)
            pred_y = pred_y.cpu()
            
            for idx in range(self.option.num_heads):
                pred_class = torch.argmax(pred_y[:,idx,:], dim=1).detach()
                accuracy_neg[idx] += (torch.sum(pred_class==y).item())/len(y)
                accuracy[idx] += (torch.sum(pred_class==y).item())/len(y)
        
        accuracy_neg /= counter
        total_counter += counter
        
        
        accuracy /= total_counter
        
        weighted_acc = (accuracy_pos+accuracy_neg)/2
        
        logging.info("============= Classifiers Evaluated =============")
        for idx in range(self.option.num_heads):
            logging.info("idx:{} true accuracy: {:.4f} false accuracy: {:.4f} total accuracy: {:.4f} weighted accuracy: {:.4f}".format(
                idx,
                accuracy_pos[idx],
                accuracy_neg[idx],
                accuracy[idx],
                weighted_acc[idx])
            )
        logging.info("===============================================")

        self.accuracy_pos = accuracy_pos
        self.accuracy_neg = accuracy_neg
        self.accuracy = accuracy
        self.weighted_accuracy = weighted_acc
        
        save_dir = os.path.join(self.save_dir, "classifier_accuracies")
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'accuracy.npy'), self.accuracy)
        np.save(os.path.join(save_dir, 'accuracy_pos.npy'), self.accuracy_pos)
        np.save(os.path.join(save_dir, 'accuracy_neg.npy'), self.accuracy_neg)
        np.save(os.path.join(save_dir, 'weighted_accuracy.npy'), self.weighted_accuracy)


