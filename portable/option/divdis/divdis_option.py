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
from portable.option.policy.intrinsic_motivation.tabular_count import TabularCount

from portable.option.sets.utils import BayesianWeighting

@gin.configurable 
class DivDisOption():
    def __init__(self,
                 use_gpu,
                 log_dir,
                 save_dir,
                 num_heads,
                 
                 policy_phi,
                 use_seed_for_initiation,
                 exp_type,
                 tabular_beta=0.0,
                 beta_distribution_alpha=100,
                 beta_distribution_beta=100,
                 video_generator=None,
                 plot_dir=None):
        
        assert len(use_gpu) == num_heads
        
        self.gpu_list = use_gpu
        self.save_dir = save_dir
        self.policy_phi = policy_phi
        self.log_dir = log_dir
        self.exp_type = exp_type
        
        self.use_seed_for_initiation = use_seed_for_initiation
        
        self.terminations = DivDisClassifier(use_gpu=use_gpu[0],
                                             head_num=num_heads,
                                             log_dir=os.path.join(log_dir, 'termination'))
        
        self.num_heads = num_heads
        self.option_steps = [0]*self.num_heads
        
        if self.use_seed_for_initiation:
            self.policies = [
                {} for _ in range(self.num_heads)
            ]
        else:
            self.policies = [
                [] for _ in range(self.num_heads)
            ]
        
        self.initiable_policies = None
        self.video_generator = video_generator
        self.make_plots = False
        
        self.confidences = BayesianWeighting(beta_distribution_alpha,
                                             beta_distribution_beta,
                                             num_heads)
        
        self.intrinsic_bonuses = [TabularCount(beta=tabular_beta) for _ in range(num_heads)]
        
        self.train_data = {}
        for idx in range(self.num_heads):
            self.train_data[idx] = []
    
    def _video_log(self, line):
        if self.video_generator is not None:
            self.video_generator.add_line(line)
    
    def _get_termination_save_path(self):
        return os.path.join(self.save_dir, 'termination')
    
    def save(self):
        os.makedirs(self._get_termination_save_path(), exist_ok=True)
        
        self.terminations.save(path=self._get_termination_save_path())
        for idx, policies in enumerate(self.policies):
            for key in policies.keys():
                policies[key].save(os.path.join(self.save_dir, "{}_{}".format(idx, key)))
        
            with open(os.path.join(self.save_dir, "{}_policy_keys.pkl".format(idx)), "wb") as f:
                pickle.dump(list(policies.keys()), f)
        
        self.confidences.save(os.path.join(self.save_dir, 'confidence.pkl'))
        with open(os.path.join(self.save_dir, "experiment_results.pkl"), 'wb') as f:
            pickle.dump(self.train_data, f)
        for idx, bonus in enumerate(self.intrinsic_bonuses):
            bonus.save(os.path.join(self.save_dir, 'bonus_{}'.format(idx)))
    
    def load(self):
        if os.path.exists(self._get_termination_save_path()):
            # print in green text
            print("\033[92m {}\033[00m" .format("Termination model loaded"))
            self.terminations.load(self._get_termination_save_path())
            for idx, policies in enumerate(self.policies):
                with open(os.path.join(self.save_dir, "{}_policy_keys.pkl".format(idx)), "rb") as f:
                    keys = pickle.load(f)
                for key in keys:
                    policies[key] = PolicyWithInitiation(use_gpu=self.gpu_list[idx],
                                                        policy_phi=self.policy_phi,
                                                        learn_initiation=(not self.use_seed_for_initiation))
                    policies[key].load(os.path.join(self.save_dir, "{}_{}".format(idx, key)))
            self.confidences.load(os.path.join(self.save_dir, 'confidence.pkl'))
            with open(os.path.join(self.save_dir, "experiment_results.pkl"), 'rb') as f:
                self.train_data = pickle.load(f)
            for idx, bonus in enumerate(self.intrinsic_bonuses):
                bonus.load(os.path.join(self.save_dir, 'bonus_{}'.format(idx)))
        else:
            # print in red text
            print("\033[91m {}\033[00m" .format("No Checkpoint found. No model has been loaded"))
    
    def add_policy(self, 
                   term_idx):
        self.policies[term_idx].append(PolicyWithInitiation(use_gpu=self.gpu_list[term_idx],
                                                            policy_phi=self.policy_phi))
    
    def find_possible_policy(self, *kwargs):
        if self.use_seed_for_initiation:
            return self._seed_possible_policies(*kwargs)
        else:
            return self._initiation_possible_policies(*kwargs)
    
    def _initiation_possible_policies(self, obs):
        policy_idxs = []
        
        for policies in self.policies:
            idxs = []
            for idx in range(len(policies)):
                if policies[idx].can_initiate(obs):
                    idxs.append(idx)
            policy_idxs.append(idxs)
        
        self.initiable_policies = policy_idxs
        
        return policy_idxs
    
    def _seed_possible_policies(self, seed):
        mask = [False]*self.num_heads
        for idx, policies in enumerate(self.policies):
            if seed in policies.keys():
                mask[idx] = True
        
        return mask
    
    def add_datafiles(self,
                      positive_files,
                      negative_files,
                      unlabelled_files):
        self.terminations.add_data(positive_files,
                                   negative_files,
                                   unlabelled_files)
    
    def add_unlabelled_data(self, data):
        self.terminations.add_unlabelled_data(data)
    
    def _get_policy(self, head_idx, option_idx):
        if self.use_seed_for_initiation:
            if option_idx not in self.policies[head_idx].keys():
                self.policies[head_idx][option_idx] = self._get_new_policy(head_idx)
            return self.policies[head_idx][option_idx], os.path.join(self.save_dir,"{}_{}".format(head_idx, option_idx))
        else:
            if len(self.initiable_policies[head_idx]) > 0:
                return self.policies[head_idx][option_idx], os.path.join(self.save_dir,"{}_{}".format(head_idx, option_idx))
            else:
                policy = self._get_new_policy(head_idx)
                self.policies[head_idx].append(policy)
                policy.store_buffer(os.path.join(self.save_dir,"{}_{}".format(head_idx, len(self.policies[head_idx]) - 1)))
                return policy, os.path.join(self.save_dir,"{}_{}".format(head_idx, len(self.policies[head_idx]) - 1))
    
    def _get_new_policy(self, head_idx):
        return PolicyWithInitiation(use_gpu=self.gpu_list[head_idx],
                                    policy_phi=self.policy_phi,
                                    learn_initiation=(not self.use_seed_for_initiation))
    
    def set_policy_save_to_disk(self, idx, policy_idx, store_buffer_bool):
        policy, _ = self._get_policy(head_idx=idx, option_idx=policy_idx)
        policy.store_buffer_to_disk = store_buffer_bool
    
    def train_policy(self, 
                     idx,
                     env,
                     state,
                     info,
                     policy_idx,
                     max_steps=1e6,
                     make_video=False,
                     perfect_term=lambda x: False):
        
        steps = 0
        rewards = []
        option_rewards = []
        extrinsic_rewards = []
        states = []
        infos = []
        
        done = False
        should_terminate = False
        
        policy, buffer_dir = self._get_policy(idx, policy_idx)
        # policy.move_to_gpu()
        # self.terminations.move_to_gpu()
        policy.load_buffer(buffer_dir)
        
        while not (done or should_terminate or (steps >= max_steps)):
            states.append(state)
            infos.append(info)
            
            action = policy.act(state)
            if make_video and self.video_generator:
                self._video_log("[option] action: {}".format(action))
                if self.exp_type == "minigrid":
                    self.video_generator.make_image(env.render())
                else:
                    self.video_generator.make_image(env.render("rgb_array"))
            
            next_state, reward, done, info = env.step(action)
            term_state = self.policy_phi(next_state).unsqueeze(0)
            pred_y = self.terminations.predict_idx(term_state, idx)
            should_terminate = torch.argmax(pred_y) == 1
            steps += 1
            self.option_steps[idx] += 1
            rewards.append(reward)
            
            if make_video:
                self._video_log("In termination: {}".format(should_terminate))
            
            if should_terminate:
                reward = 1
                extrinsic_rewards.append(1)
            else:
                extrinsic_rewards.append(0)
                reward = self.intrinsic_bonuses[idx].get_bonus(info["player_pos"])
            
            policy.observe(state,
                           action,
                           reward,
                           next_state,
                           done or should_terminate)
            
            option_rewards.append(reward)

            state = next_state
        
        if not self.use_seed_for_initiation:
            if should_terminate:
                policy.add_data_initiation(positive_examples=states)
            else:
                policy.add_data_initiation(negative_examples=states)
            policy.add_context_examples(states)
        
        # policy.move_to_cpu()
        # self.terminations.move_to_cpu()
        policy.store_buffer(buffer_dir)
        policy.end_skill(sum(extrinsic_rewards))
        
        self.train_data[int(idx)].append({
            "head_idx": idx,
            "frames": self.option_steps[idx],
            "policy_idx": policy_idx,
            "rewards": rewards,
            "option_length": steps,
            "option_rewards": option_rewards,
            "extrinsic_rewards": extrinsic_rewards
        })
        
        return state, info, done, steps, rewards, option_rewards, states, infos
    
    def plot_term_state(self, 
                        state, 
                        img,
                        next_state,
                        next_img, 
                        idx, 
                        success,
                        pred_y):
        x = 0
        pred_y = pred_y.squeeze().cpu().numpy()
        if success is True:
            plot_dir = os.path.join(self.plot_dir, "term_states", str(idx))
        else:
            plot_dir = os.path.join(self.plot_dir, "missed_states", str(idx))
        os.makedirs(plot_dir, exist_ok=True)
        while os.path.exists(os.path.join(plot_dir, "{}.png".format(x))):
            x += 1
        plot_file = os.path.join(plot_dir, "{}.png".format(x))
        
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.imshow(img)
        ax1.set_title(f"{state[:len(state)//2]}\n{state[len(state)//2:]}", size=8)
        ax1.axis('off')
        ax2.imshow(next_img)
        ax2.set_title(f"{next_state[:len(state)//2]}\n{next_state[len(state)//2:]}", size=8)
        ax2.axis('off')
        fig.suptitle("Pred: [not term {:.4f}, is term{:.4f}]".format(
            pred_y[0],
            pred_y[1]
        ))
        
        fig.savefig(plot_file)
        plt.close(fig)
        
    
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
                    seed,
                    make_video=True,
                    max_steps=1e6):
        
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
        buffer_dir = os.path.join(self.save_dir,"{}_{}".format(idx, seed))
        # policy.move_to_gpu()
        # self.terminations.move_to_gpu()
        policy.load_buffer(buffer_dir)
        
        with evaluating(policy):
            while not (done or should_terminate):
                states.append(state)
                infos.append(info)
                
                action = policy.act(state)
                self._video_log("action: {}".format(action))
                self._video_log("State representation: {}".format(state))
                if self.video_generator is not None:
                    if self.exp_type == "minigrid":
                        img = env.render()
                    else:
                        img = env.render("rgb_array")
                    self.video_generator.make_image(img)
                
                next_state, reward, done, info = env.step(action)
                if type(next_state) is np.ndarray:
                    next_state = torch.from_numpy(next_state)
                term_state = self.policy_phi(next_state).unsqueeze(0)
                should_terminate = torch.argmax(self.terminations.predict_idx(term_state, idx)) == 1
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
            
            if make_video:
                if should_terminate:
                    self._video_log("policy hit termination")
                if done:
                    self._video_log("environment terminated")
                if steps >= max_steps:
                    self._video_log("option timed out")
            
            if self.video_generator is not None and make_video:
                if self.exp_type == "minigrid":
                    self.video_generator.make_image(env.render())
                else:
                    self.video_generator.make_image(env.render("rgb_array"))
            
            # policy.move_to_cpu()
            # self.terminations.move_to_cpu()
            policy.store_buffer(buffer_dir)
            
            return state, info, steps, rewards, option_rewards, states, infos
    
    def evaluate_states(self,
                        idx,
                        states,
                        seed):
        actions, q_vals = self.policies[idx][seed].batch_act(states)
        return actions, q_vals
    
    def get_confidences(self):
        return self.confidences.weights()
    
    def update_confidences(self,
                           update):
        self.confidences.update_successes(update)














