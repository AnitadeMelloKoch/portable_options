import logging 
import os 
import numpy as np 
import gin 
import random 
import torch 
import pickle
from torch.utils.tensorboard import SummaryWriter 

from portable.option.divdis.policy.policy_and_initiation import PolicyWithInitiation
from portable.option.divdis.policy.skill_ppo import SkillPPO
from portable.option.policy.agents import evaluating
import matplotlib.pyplot as plt 
from collections import deque

from portable.option.sets.utils import BayesianWeighting
from portable.option.policy.intrinsic_motivation.tabular_count import TabularCount

@gin.configurable 
class DivDisMockOption():
    def __init__(self,
                 use_gpu,
                 log_dir,
                 save_dir,
                 terminations,
                 
                 policy_phi,
                 use_seed_for_initiation,
                 exp_type,
                 tabular_beta=0.0,
                 beta_distribution_alpha=100,
                 beta_distribution_beta=100,
                 video_generator=None,
                 plot_dir=None):
        
        if type(use_gpu) is list:
            assert len(use_gpu) == len(terminations)
        else:
            use_gpu = [use_gpu]*len(terminations)
        
        self.gpu_list = use_gpu
        self.save_dir = save_dir
        self.policy_phi = policy_phi
        self.log_dir = log_dir
        self.exp_type = exp_type
        # this is for debugging. By using seed for initiation we do not need to 
        # learn the initiation set and are assuming the initiation set is the seed
        self.use_seed_for_initiation = use_seed_for_initiation
        
        self.terminations = terminations
        
        self.num_heads = len(terminations)
        
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
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        if plot_dir is not None:
            self.make_plots = True
            self.plot_dir = plot_dir
            self.term_states = []
            self.missed_term_states = []
        
        self.train_rewards = deque(maxlen=200)
        self.option_steps = [0]*self.num_heads
        
        self.confidences = BayesianWeighting(beta_distribution_alpha,
                                             beta_distribution_beta,
                                             self.num_heads)
        
        self.intrinsic_bonuses = [TabularCount(beta=tabular_beta) for _ in range(self.num_heads)]
        
        self.train_data = {}
        self.writer = SummaryWriter(log_dir=log_dir)
        
        for idx in range(self.num_heads):
            self.train_data[idx] = []
    
    def _video_log(self, line):
        if self.video_generator is not None:
            self.video_generator.add_line(line)
    
    def _get_termination_save_path(self):
        return os.path.join(self.save_dir, 'termination')
    
    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        for idx, policies in enumerate(self.policies):
            for key in policies.keys():
                policies[key].save(os.path.join(self.save_dir, "{}_{}".format(idx, key)))
        
            with open(os.path.join(self.save_dir, "{}_policy_keys.pkl".format(idx)), "wb") as f:
                pickle.dump(list(policies.keys()), f)
        with open(os.path.join(self.save_dir, "experiment_results.pkl"), 'wb') as f:
            pickle.dump(self.train_data, f)
        for idx, bonus in enumerate(self.intrinsic_bonuses):
            bonus.save(os.path.join(self.save_dir, 'bonus_{}'.format(idx)))
    
    def load(self):
        for idx, policies in enumerate(self.policies):
            with open(os.path.join(self.save_dir, "{}_policy_keys.pkl".format(idx)), "rb") as f:
                keys = pickle.load(f)
            for key in keys:
                policies[key] = PolicyWithInitiation(use_gpu=self.gpu_list[idx],
                                                     policy_phi=self.policy_phi)
                policies[key].load(os.path.join(self.save_dir, "{}_{}".format(idx, key)))
        with open(os.path.join(self.save_dir, "experiment_results.pkl"), 'rb') as f:
            self.train_data = pickle.load(f)
        for idx, bonus in enumerate(self.intrinsic_bonuses):
            bonus.load(os.path.join(self.save_dir, 'bonus_{}'.format(idx)))
        
    
    def reset_classifiers(self):
        return
    
    def reset_dataset(self):
        return
    
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
        # not needed for mock terminations
        # should return and do nothing
        return
    
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
                                    policy_phi=self.policy_phi)
    
    def set_policy_save_to_disk(self, idx, policy_idx, store_buffer_bool):
        policy, _ = self._get_policy(head_idx=idx, option_idx=policy_idx)
        policy.store_buffer_to_disk = store_buffer_bool
    
    def check_termination(self, idx, state, env):
        return self.terminations[idx](state, env)
    
    def train_policy(self, 
                     idx,
                     env,
                     state,
                     info,
                     policy_idx,
                     max_steps=1e6,
                     make_video=False):
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
        # policy.load_buffer(buffer_dir)
        
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
                    
            
            # print("action", action)
            # print(state)
            # plt.imshow(np.transpose(state.cpu().numpy(), (1,2,0)))
            # plt.show(block=False)
            # input("continue")
            
            next_state, reward, done, info = env.step(action)
            
            if type(next_state) is np.ndarray:
                next_state = torch.from_numpy(next_state)
            
            self.option_steps[idx] += 1
            
            should_terminate = self.terminations[idx](state,
                                                      env)
            
            if make_video:
                self._video_log("In termination: {}".format(should_terminate))
                # if policy.initiation.is_initialized():
                #     self._video_log("Initiation: {}".format(policy.initiation.pessimistic_predict(next_state)))
            
            steps += 1
            rewards.append(reward)
            
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
        # policy.store_buffer(buffer_dir)
        policy.end_skill(sum(option_rewards))
        
        self.train_data[int(idx)].append({
            "head_idx": idx,
            "frames": self.option_steps[idx],
            "policy_idx": policy_idx,
            "rewards": rewards,
            "option_length": steps,
            "option_rewards": option_rewards,
            "extrinsic_rewards": extrinsic_rewards
        })
        
        self.writer.add_scalar('option_length/{}'.format(policy_idx), steps, policy.option_runs)
        self.writer.add_scalar('intrinsic_reward/{}'.format(policy_idx), sum(option_rewards), policy.option_runs)
        self.writer.add_scalar('option_reward/{}'.format(policy_idx), sum(extrinsic_rewards), policy.option_runs)
        
        return state, info, done, steps, rewards, extrinsic_rewards, states, infos
    
    def bootstrap_policy(self,
                         idx,
                         envs,
                         max_steps,
                         min_performance,
                         seed,
                         agent_start_positions=[]):
        total_steps = 0
        train_rewards = deque(maxlen=50)
        eval_rewards = deque(maxlen=200)
        episode = 0
        
        # self.set_policy_save_to_disk(idx, seed, False)
        
        while total_steps < max_steps:
            env = random.choice(envs)
            rand_num = np.random.randint(low=0, high=50)
            if len(agent_start_positions) > 0:
                agent_position = random.choice(agent_start_positions)
            else:
                agent_position = None
            obs, info = env.reset(agent_reposition_attempts=rand_num,
                                  random_start=True,
                                  agent_position=agent_position)
            
            if type(obs) is np.ndarray:
                obs = torch.from_numpy(obs)
            
            _, _, _, steps, _, rewards, _, _ = self.train_policy(idx,
                                                              env,
                                                              obs,
                                                              info,
                                                              seed)
            
            total_steps += steps
            train_rewards.append(sum(rewards))
            # eval_run_rewards = self._get_eval_performance(idx,
            #                                               envs,
            #                                               1,
            #                                               seed)
            # eval_rewards.append(sum(eval_run_rewards))
            
            if episode % 1 == 0:
                logging.info("idx {} steps: {} average train reward: {} average eval reward {}".format(idx,
                                                                          total_steps,
                                                                          np.mean(train_rewards),
                                                                          np.mean(eval_rewards)))
            episode += 1
            
            # if total_steps > 200000 and np.mean(eval_rewards) > min_performance:
            #     logging.info("idx {} reached required performance with average reward: {} at step {}".format(idx,
            #                                                                                                  np.mean(eval_rewards),
            #                                                                                                  total_steps))
            #     break
        
        self.save()
        # self.set_policy_save_to_disk(idx, seed, True)
        
    
    def _get_eval_performance(self,
                              idx,
                              envs,
                              num_runs,
                              seed):
        option_rewards = []
        for _ in range(num_runs):
            env = random.choice(envs)
            rand_num = np.random.randint(low=0, high=100)
            obs, info = env.reset(agent_reposition_attempts=rand_num,
                                  random_start=True)
            
            _, _, _, _, _, run_rewards, _, _, = self.eval_policy(idx,
                                                                 env,
                                                                 obs,
                                                                 info,
                                                                 seed,
                                                                 make_video=False,
                                                                 max_steps=500)
            option_rewards.append(sum(run_rewards))
        
        
        return option_rewards
    
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
        
        # if seed not in self.policies[idx]:
        #     raise Exception("Policy has not been initialized. Train policy before evaluating")
        
        policy = self.policies[idx][seed]
        buffer_dir = os.path.join(self.save_dir,"{}_{}".format(idx, seed))
        # policy.move_to_gpu()
        # policy.load_buffer(buffer_dir)
        
        with evaluating(policy):
            while not (done or should_terminate or (steps >= max_steps)):
                states.append(state)
                infos.append(info)
                
                action = policy.act(state)
                if make_video:
                    self._video_log("[option] action: {}".format(action))
                if self.video_generator is not None and make_video:
                    if self.exp_type == "minigrid":
                        self.video_generator.make_image(env.render())
                    else:
                        self.video_generator.make_image(env.render("rgb_array"))
                
                next_state, reward, done, info = env.step(action)
                should_terminate = self.terminations[idx](state,
                                                          env)
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
            # policy.store_buffer(buffer_dir)
            
            self.writer.add_scalar('eval_option_length/{}'.format(idx), steps, policy.option_runs)
            self.writer.add_scalar('eval_intrinsic_reward/{}'.format(idx), sum(option_rewards), policy.option_runs)
            
            return state, info, done, steps, rewards, option_rewards, states, infos
    
    def evaluate_states(self,
                        idx,
                        states,
                        seed):
        # self.policies[idx][seed].move_to_gpu()
        actions, q_vals = self.policies[idx][seed].batch_act(states)
        # self.policies[idx][seed].move_to_cpu()
        return actions, q_vals
    
    def get_confidences(self):
        return self.confidences.weights()
    
    def update_confidences(self,
                           update):
        self.confidences.update_successes(update)
    