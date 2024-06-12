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
import pickle

from portable.option.divdis.divdis_mock_option import DivDisMockOption
from portable.option.divdis.divdis_option import DivDisOption
from portable.option.divdis.global_option import GlobalOption
from experiments.experiment_logger import VideoGenerator
from portable.agent.model.ppo import ActionPPO
from portable.agent.model.maskable_ppo import MaskablePPOAgent
import math
from portable.option.memory import SetDataset

OPTION_TYPES = ["mock", "divdis"]

@gin.configurable
class DivDisMetaExperiment():
    def __init__(self,
                 base_dir,
                 experiment_name,
                 seed,
                 option_policy_phi,
                 agent_phi,
                 use_gpu,
                 option_type,
                 num_options,
                 num_primitive_actions,
                 add_unlabelled_data=False,
                 gpu_list=[0],
                 start_epsilon=0.0,
                 end_epsilon=0.0,
                 decay_steps=5e5,
                 use_global_option=False,
                 option_timeout=50,
                 action_policy=None,
                 action_vf=None,
                 action_model=None,
                 classifier_epochs=50,
                 terminations=[],
                 option_head_num=4,
                 discount_rate=0.9,
                 make_videos=False):
        
        assert option_type in OPTION_TYPES
        
        if action_model is None:
            self.exp = "minigrid"
        else:
            self.exp = "monte"
        
        self.name = experiment_name
        self.seed = seed 
        self.use_gpu = use_gpu
        self.option_type = option_type
        self.classifier_epochs = classifier_epochs
        self.option_timeout = option_timeout
        self.use_global_option = use_global_option
        self.add_unlabelled_data = add_unlabelled_data
        
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_steps = decay_steps
        self.epsilon = start_epsilon
        
        self.base_dir = os.path.join(base_dir, experiment_name, str(seed))
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.save_dir = os.path.join(self.base_dir, 'checkpoints')
        self.plot_dir = os.path.join(self.base_dir, 'plots')
        
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        self.decisions = 0
        
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
        
        self.meta_agent = ActionPPO(use_gpu=gpu_list[-1],
                                           policy=action_policy,
                                           value_function=action_vf,
                                           model=action_model,
                                           phi=agent_phi)
        
        self.options = []
        self.num_options = num_options
        self.num_primitive_actions = num_primitive_actions
        
        self._cumulative_discount_vector = np.array(
            [math.pow(discount_rate, n) for n in range(100)]
        )
        
        gpu_assign_list = self._assign_gpus(self.use_gpu, num_options*option_head_num, gpu_list)
        
        if self.use_global_option:
            gpu_assign_list = self._assign_gpus(self.use_gpu, (num_options*option_head_num)+1, gpu_list)
            self.global_option = GlobalOption(use_gpu=gpu_assign_list[-1],
                                              log_dir=os.path.join(self.log_dir),
                                              save_dir=os.path.join(self.save_dir),
                                              policy_phi=option_policy_phi)
        
        if self.option_type == "mock":
            assert len(terminations) == num_options
            for idx, termination_list in enumerate(terminations):
                self.options.append(DivDisMockOption(use_gpu=gpu_assign_list[idx*option_head_num: (idx+1)*option_head_num],
                                                    log_dir=os.path.join(self.log_dir, "option_{}".format(idx)),
                                                    save_dir=os.path.join(self.save_dir, "option_{}".format(idx)),
                                                    terminations=termination_list,
                                                    policy_phi=option_policy_phi,
                                                    video_generator=self.video_generator,
                                                    plot_dir=os.path.join(self.plot_dir, "option_{}".format(idx)),
                                                    use_seed_for_initiation=True))
            if len(self.options) > 0:
                self.num_heads = self.options[0].num_heads
            else:
                self.num_heads = 0
        
        elif self.option_type == "divdis":
            self.num_heads = option_head_num
            for idx in range(self.num_options):
                self.options.append(DivDisOption(use_gpu=gpu_assign_list[idx*option_head_num: (idx+1)*option_head_num],
                                                 log_dir=os.path.join(self.log_dir, "option_{}".format(idx)),
                                                 save_dir=os.path.join(self.save_dir, "option_{}".format(idx)),
                                                 num_heads=option_head_num,
                                                 policy_phi=option_policy_phi,
                                                 video_generator=self.video_generator,
                                                 plot_dir=os.path.join(self.plot_dir, "option_{}".format(idx)),
                                                 use_seed_for_initiation=True))
        
        self.total_actions = self.num_primitive_actions + (self.num_options*self.num_heads)
        self.gamma = discount_rate
        
        self.experiment_data = []
        self.episode_data = []
        
    @staticmethod
    def _assign_gpus(use_gpu, num_models, gpu_list):
        assign_list_idx = [
            idx%len(gpu_list) for idx in range(num_models)
        ]
        if use_gpu is False:
            assign_list_idx = [-1]*num_models
        return assign_list_idx
    
    def save(self):
        for option in self.options:
            option.save()
        with open(os.path.join(self.save_dir, "experiment_results.pkl"), 'wb') as f:
            pickle.dump(self.experiment_data, f)
        
        with open(os.path.join(self.save_dir, "episode_results.pkl"), 'wb') as f:
            pickle.dump(self.episode_data, f)
        
        if self.use_global_option:
            self.global_option.save()
        
        np.save(os.path.join(self.save_dir, "decisions.npy"),self.decisions)
    
    def load(self):
        for option in self.options:
            option.load()
        with open(os.path.join(self.save_dir, "experiment_results.pkl"), 'rb') as f:
            self.experiment_data = pickle.load(f)
        
        with open(os.path.join(self.save_dir, "episode_results.pkl"), 'rb') as f:
            self.episode_data = pickle.load(f)
        
        if self.use_global_option:
            self.global_option.load()
        
        self.decisions = np.load(os.path.join(self.save_dir, "decisions.npy"))
    
    def add_datafiles(self,
                      positive_files,
                      negative_files,
                      unlabelled_files):
        assert len(positive_files) == self.num_options
        assert len(negative_files) == self.num_options
        assert len(unlabelled_files) == self.num_options
        
        for idx in range(len(positive_files)):
            self.options[idx].add_datafiles(positive_files[idx],
                                            negative_files[idx],
                                            unlabelled_files[idx])
    
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
    
    def train_option_classifiers(self, epochs=None):
        if epochs is None:
            epochs = self.classifier_epochs
        for idx in range(self.num_options):
            self.options[idx].terminations.train(epochs)
    
    def get_masks_from_seed(self,
                            seed):
        action_mask = [True]*(self.num_primitive_actions)
        
        for option in self.options:
            option_mask = option.find_possible_policy(seed)
            action_mask.extend(option_mask)
        
        return action_mask
    
    def get_termination_masks(self, state, env):
        masks = []
        if self.use_global_option:
            masks.append(torch.tensor(False, dtype=bool))
        for option in self.options:
            for idx in range(option.num_heads):
                term = option.check_termination(idx, state, env)
                if type(term) is not torch.Tensor:
                    term = torch.tensor(term, dtype=bool)
                term = term.cpu()
                masks.append(term)
        
        return masks
    
    def act(self, obs, mask):
        action, q_vals = self.meta_agent.act(obs, mask)
        
        return action, q_vals
    
    def observe(self, 
                obs,
                mask, 
                rewards, 
                done):
        
        if len(rewards) > len(self._cumulative_discount_vector):
            self._cumulative_discount_vector = np.array(
                [math.pow(self.gamma, n) for n in range(len(rewards))]
            )
        
        reward = np.sum(self._cumulative_discount_vector[:len(rewards)]*rewards)
        
        self.meta_agent.observe(obs,
                                reward,
                                done,
                                done)
    
    def save_image(self, env):
        if self.video_generator is not None:
            if self.exp == "minigrid":
                img = env.render()
            else:
                img = env.render("rgb_array")
            self.video_generator.make_image(img)
    
    def _compute_epsilon(self):
        if self.decisions > self.decay_steps:
            return self.end_epsilon
        else:
            epsilon_diff = self.end_epsilon - self.start_epsilon
            return self.start_epsilon + epsilon_diff * (self.decisions / self.decay_steps)

    def select_action(self, action):
        if self.num_primitive_actions == 0:
            return action
        
        self.epsilon = self._compute_epsilon()
        if np.random.rand() < self.epsilon:
            if np.random.rand() < 0.5:
                return np.random.randint(self.num_primitive_actions)
            else:
                return np.random.randint(self.num_primitive_actions, self.total_actions)
        else:
            return action

    def train_meta_agent(self,
                         env,
                         seed,
                         max_steps,
                         min_performance=1.0):
        total_steps = 0
        episode_rewards = deque(maxlen=200)
        episode = 0
        undiscounted_rewards = []
        
        while total_steps < max_steps:
            undiscounted_reward = 0         # episode reward
            done = False
            
            if self.video_generator is not None:
                self.video_generator.episode_start()
            
            obs, info = env.reset()
            
            while not done:
                self.save_image(env)
                if type(obs) == np.ndarray:
                    obs = torch.from_numpy(obs).float()
                action_mask = self.get_termination_masks(obs, env)
                action, q_vals = self.act(obs, action_mask)
                
                self._video_log("action: {}".format(action))
                self._video_log("action q vals: {}".format(q_vals))
                
                if self.use_global_option:
                    if action == 0:
                        next_obs, reward, done, info, steps = self.global_option.train_policy(env=env,
                                                                                    info=info,
                                                                                    make_video=False,
                                                                                    obs=obs)
                    else:
                        action = action - 1
                
                if action < self.num_primitive_actions:
                    next_obs, reward, done, info = env.step(action)
                    undiscounted_reward += reward
                    rewards = [reward]
                    steps = 1
                else:
                    action_offset = action-self.num_primitive_actions
                    option_num = int(action_offset/self.num_heads)
                    option_head = action_offset%self.num_heads
                    next_obs, info, done, steps, rewards, _, states, _ = self.options[option_num].train_policy(option_head,
                                                                                                            env,
                                                                                                            obs,
                                                                                                            info,
                                                                                                            seed,
                                                                                                            max_steps=self.option_timeout,
                                                                                                            make_video=False)
                undiscounted_reward += np.sum(rewards)
                self.decisions += 1
                total_steps += steps
                
                if self.add_unlabelled_data is True:
                    self.options[option_num].add_unlabelled_data(states)
                
                self.experiment_data.append({
                    "meta_step": self.decisions,
                    "option_length": steps,
                    "option_rewards": rewards,
                    "frames": total_steps
                })
                
                self.observe(obs,
                             action_mask,
                             rewards,
                             done)
                obs = next_obs
            if self.add_unlabelled_data is True:
                self.train_option_classifiers(1)
            
            logging.info("Episode {} total steps: {} decisions: {}  average undiscounted reward: {}".format(episode,
                                                                                     total_steps,
                                                                                     self.decisions,  
                                                                                     np.mean(episode_rewards)))
            
            if (undiscounted_reward > 0 or episode%10==0) and self.video_generator is not None:
                self.video_generator.episode_end("episode_{}".format(episode))
            
            undiscounted_rewards.append(undiscounted_reward)
            episode += 1
            episode_rewards.append(undiscounted_rewards)
            
            self.episode_data.append({
                "episode": episode,
                "episode_rewards": undiscounted_reward,
                "frames": total_steps
            })
            
            self.writer.add_scalar('episode_rewards', undiscounted_reward, total_steps)
            
            # self.plot_learning_curve(episode_rewards)
            
            if episode % 50 == 0:
                self.meta_agent.save(os.path.join(self.save_dir, "action_agent"))
                self.save()
            
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
        
        plt.close(fig)
        
    
    def eval_meta_agent(self,
                        env,
                        seed,
                        num_runs):
        undiscounted_rewards = []
        
        with self.meta_agent.agent.eval_mode():
            for run in range(num_runs):
                total_steps = 0
                undiscounted_reward = 0
                done = False
                if self.video_generator is not None:
                    self.video_generator.episode_start()
                obs, info = env.reset()
                while not done:
                    self.save_image(env)
                    if type(obs) == np.ndarray:
                        obs = torch.from_numpy(obs).float()
                    action_mask = self.get_termination_masks(obs, env)
                    action, q_vals = self.act(obs, action_mask)
                    
                    self._video_log("[meta] action: {}".format(action))
                    self._video_log("[meta] action q values")
                    for idx in range(len(q_vals[0])):
                        self._video_log("[meta] action {} value {}".format(idx, q_vals[0][idx]))
                    
                    if self.use_global_option:
                        if action == 0:
                            next_obs, reward, done, info, steps = self.global_option.eval_policy(env=env,
                                                                                                 info=info,
                                                                                                 obs=obs)
                        else:
                            action = action - 1
                    
                    if action < self.num_primitive_actions:
                        next_obs, reward, done, info = env.step(action)
                        undiscounted_reward += reward
                        rewards = [reward]
                        total_steps += 1
                        steps = 1
                    else:
                        action_offset = action-self.num_primitive_actions
                        option_num = int(action_offset/self.num_heads)
                        option_head = action_offset%self.num_heads
                        self._video_log("[meta] selected option {}".format(action-self.num_primitive_actions))
                        next_obs, info, done, steps, rewards, _, _, _ = self.options[option_num].eval_policy(option_head,
                                                                                                             env,
                                                                                                             obs,
                                                                                                             info,
                                                                                                             seed)
                        undiscounted_reward += np.sum(rewards)
                        total_steps += steps
                    
                        self.observe(obs,
                                     action_mask,
                                     rewards,
                                     done)
                        obs = next_obs
                
                logging.info("Eval {} total steps: {} undiscounted reward: {}".format(run,
                                                                                      total_steps,
                                                                                      undiscounted_reward))
            
                if self.video_generator is not None:
                    self.video_generator.episode_end("eval_{}".format(run))
                
                undiscounted_rewards.append(undiscounted_reward)
    
    def test_classifiers(self,
                         test_positive_files,
                         test_negative_files):
        assert len(test_positive_files) == self.num_options
        assert len(test_negative_files) == self.num_options
        
        self.accuracy_pos = []
        self.accuracy_neg = []
        self.weighted_accuracy = []
        self.accuracy = []
        
        for option_idx in range(self.num_options):
            dataset_positive = SetDataset(max_size=1e6,
                                          batchsize=64)
            dataset_negative = SetDataset(max_size=1e6,
                                          batchsize=64)
            
            dataset_positive.add_true_files(test_positive_files[option_idx])
            dataset_negative.add_false_files(test_negative_files[option_idx])
            
            counter = 0
            accuracy = np.zeros(self.num_heads)
            accuracy_pos = np.zeros(self.num_heads)
            accuracy_neg = np.zeros(self.num_heads)
            
            for _ in range(dataset_positive.num_batches):
                counter += 1
                x, y = dataset_positive.get_batch()
                pred_y, _ = self.options[option_idx].terminations.predict(x)
                pred_y = pred_y.cpu()
                
                for idx in range(self.num_heads):
                    pred_class = torch.argmax(pred_y[:,idx,:], dim=1).detach()
                    accuracy_pos[idx] += (torch.sum(pred_class==y).item())/len(y)
                    accuracy[idx] += (torch.sum(pred_class==y).item())/len(y)
            
            accuracy_pos /= counter
            
            total_counter = counter
            counter = 0
            
            for _ in range(dataset_negative.num_batches):
                counter += 1
                x, y = dataset_negative.get_batch()
                pred_y, _ = self.options[option_idx].terminations.predict(x)
                pred_y = pred_y.cpu()
                
                for idx in range(self.num_heads):
                    pred_class = torch.argmax(pred_y[:,idx,:], dim=1).detach()
                    accuracy_neg[idx] += (torch.sum(pred_class==y).item())/len(y)
                    accuracy[idx] += (torch.sum(pred_class==y).item())/len(y)
            
            accuracy_neg /= counter
            total_counter += counter
            
            
            accuracy /= total_counter
            
            weighted_acc = (accuracy_pos+accuracy_neg)/2
            
            logging.info("============= Option {} Evaluated =============".format(option_idx))
            for idx in range(self.num_heads):
                logging.info("idx:{} true accuracy: {:.4f} false accuracy: {:.4f} total accuracy: {:.4f} weighted accuracy: {:.4f}".format(
                    idx,
                    accuracy_pos[idx],
                    accuracy_neg[idx],
                    accuracy[idx],
                    weighted_acc[idx])
                )
            logging.info("===============================================")
            
            self.accuracy.append(accuracy)
            self.accuracy_neg.append(accuracy_neg)
            self.accuracy_pos.append(accuracy_pos)
            self.weighted_accuracy.append(weighted_acc)
        
        save_dir = os.path.join(self.save_dir, "classifier_accuracies")
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'accuracy.npy'), self.accuracy)
        np.save(os.path.join(save_dir, 'accuracy_pos.npy'), self.accuracy_pos)
        np.save(os.path.join(save_dir, 'accuracy_neg.npy'), self.accuracy_neg)
        np.save(os.path.join(save_dir, 'weighted_accuracy.npy'), self.weighted_accuracy)
    
    
    













