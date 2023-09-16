import logging
import os 
import numpy as np 
import gin 
import random 
from portable.option.policy.agents import evaluating 
from operator import countOf
from collections import deque
import math
import torch

from portable.option.sets import AttentionSet
from portable.option.policy.agents import EnsembleAgent

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

@gin.configurable
class AttentionOption():
    def __init__(self,
                 use_gpu,
                 log_dir,
                 save_dir,
                 
                 markov_option_builder,
                 initiation_vote_threshold,
                 termination_vote_threshold,
                 policy_phi,
                 prioritized_replay_anneal_steps,
                 embedding,
                 
                 policy_warmup_steps,
                 policy_batchsize,
                 policy_buffer_length,
                 policy_update_interval,
                 q_target_update_interval,
                 policy_learning_rate,
                 final_epsilon,
                 final_exploration_frames,
                 discount_rate,
                 policy_attention_module_num,
                 num_actions,
                 policy_c,
                 
                 initiation_beta_distribution_alpha,
                 initiation_beta_distribution_beta,
                 initiation_attention_module_num,
                 initiation_lr,
                 initiation_dataset_maxsize,
                 
                 termination_beta_distribution_alpha,
                 termination_beta_distribution_beta,
                 termination_attention_module_num,
                 termination_lr,
                 termination_dataset_maxsize,
                 
                 min_interactions,
                 min_success_rate,
                 timeout,
                 min_option_length,
                 original_initiation_function=None,
                 dataset_transform_function=None):
        
        self._cumulative_discount_vector = np.array(
            [math.pow(discount_rate, n) for n in range(timeout)]
        )
        
        summary_writer = SummaryWriter(log_dir=log_dir)
        self.use_gpu = use_gpu
        
        self.policy = EnsembleAgent(use_gpu=use_gpu,
                                    warmup_steps=policy_warmup_steps,
                                    batch_size=policy_batchsize,
                                    phi=policy_phi,
                                    prioritized_replay_anneal_steps=prioritized_replay_anneal_steps,
                                    embedding=embedding,
                                    buffer_length=policy_buffer_length,
                                    update_interval=policy_update_interval,
                                    q_target_update_interval=q_target_update_interval,
                                    learning_rate=policy_learning_rate,
                                    final_epsilon=final_epsilon,
                                    final_exploration_frames=final_exploration_frames,
                                    discount_rate=discount_rate,
                                    num_modules=policy_attention_module_num,
                                    num_actions=num_actions,
                                    c=policy_c,
                                    summary_writer=summary_writer)

        self.initiation = AttentionSet(use_gpu=use_gpu,
                                       embedding=embedding,
                                       log_dir=log_dir,
                                       vote_threshold=initiation_vote_threshold,
                                       beta_distribution_alpha=initiation_beta_distribution_alpha,
                                       beta_distribution_beta=initiation_beta_distribution_beta,
                                       learning_rate=initiation_lr,
                                       dataset_max_size=initiation_dataset_maxsize,
                                       attention_module_num=initiation_attention_module_num,
                                       model_name="initiation",
                                       summary_writer=summary_writer,
                                       padding_func=dataset_transform_function)
        self.initiation.move_to_gpu()
        self.termination = AttentionSet(use_gpu=use_gpu,
                                        embedding=embedding,
                                        log_dir=log_dir,
                                        vote_threshold=termination_vote_threshold,
                                        beta_distribution_alpha=termination_beta_distribution_alpha,
                                        beta_distribution_beta=termination_beta_distribution_beta,
                                        learning_rate=termination_lr,
                                        dataset_max_size=termination_dataset_maxsize,
                                        attention_module_num=termination_attention_module_num,
                                        model_name="termination",
                                        summary_writer=summary_writer,
                                        padding_func=dataset_transform_function)
        
        # build markov option
        self.markov_option_builder = markov_option_builder
        self.markov_instantiations = []
        self.attempted_instantiations = []
        self.option_timeout = timeout
        self.min_interactions = min_interactions
        self.min_option_length = min_option_length
        self.min_success_rate = min_success_rate
        self.original_initiation_function = original_initiation_function
        self.markov_idx = None
        
        self.log_dir = log_dir
        self.save_dir = save_dir
        os.makedirs(log_dir, exist_ok=True)
        
        policy_path, _, _, _ = self._get_save_paths()
        os.makedirs(policy_path, exist_ok=True)
        self.policy_buffer_save_file = os.path.join(policy_path, 'memory_buffer.pkl')
        
    def _get_save_paths(self):
        policy = os.path.join(self.save_dir, 'policy')
        initiation = os.path.join(self.save_dir, 'initiation')
        termination = os.path.join(self.save_dir, 'termination')
        markov = os.path.join(self.save_dir, 'markov')

        return policy, initiation, termination, markov

    def save(self):
        policy_path, initiation_path, termination_path, markov_path = self._get_save_paths()
        
        os.makedirs(policy_path, exist_ok=True)
        os.makedirs(initiation_path, exist_ok=True)
        os.makedirs(termination_path, exist_ok=True)
        
        self.policy.save(policy_path)
        self.initiation.save(initiation_path)
        self.termination.save(termination_path)

        for idx, instance in enumerate(self.markov_instantiations):
            save_path = os.path.join(markov_path, str(idx))
            instance.save(save_path)

        logger.info("[option] Saving option to path: {}".format(self.save_dir))

    def load(self):
        policy_path, initiation_path, termination_path, markov_path = self._get_save_paths()

        if os.path.exists(os.path.join(policy_path, 'agent.pkl')):
            self.policy = self.policy.load(os.path.join(policy_path, 'agent.pkl'))
            
        self.initiation.load(initiation_path)
        self.termination.load(termination_path)

        for idx, instance in self.markov_instantiations:
            save_path = os.path.join(markov_path, str(idx))
            if os.path.exists(save_path):
                instance.load(save_path)

        logger.info("[option] Loading option from path: {}".format(self.save_dir))

    @staticmethod
    def intersection(list_a, list_b):
        return [i for i in list_a if countOf(list_b,i) > 0]

    @staticmethod
    def disjunctive_union(list_a, list_b):
        return [i for i in list_a if countOf(list_b,i) == 0]
    
    def create_instance(self,
                        states,
                        termination_state,
                        false_states):
        ## TODO
        # how to check if we already have instance?
        
        path = os.path.join(self.save_dir, 'markov', str(len(self.markov_instantiations)))
        os.makedirs(path, exist_ok=True)
        file = os.path.join(path, 'memory_buffer.pkl')
        
        self.markov_instantiations.append(
            self.markov_option_builder(
                states=states,
                false_states=false_states,
                termination_state=termination_state,
                initial_policy=self.policy,
                initiation_votes=self.initiation.votes,
                termination_votes=self.termination.votes,
                use_gpu=self.use_gpu,
                save_file=file
            )
        )
    
    def can_initiate(self,
                     state):
        logger.info("[portable option initiation] Checking if option can initiate")
        if type(state) is np.ndarray:
            state = torch.from_numpy(state).float()
        
        vote_global, _, votes, conf = self.initiation.vote(state)
        
        local_vote = False
        for idx in range(len(self.markov_instantiations)):
            prediction = self.markov_instantiations[idx].can_initiate(state)
            
            if prediction == True:
                self.markov_idx = idx
                local_vote = True
        
        return vote_global, local_vote
    
    def run(self,
            env,
            state,
            info,
            eval,
            false_states):
        
        if type(state) is np.ndarray:
            state = torch.from_numpy(state).float()
        global_vote, markov_vote = self.can_initiate(state)
        
        if global_vote and not markov_vote:
            return self._portable_run(env,
                                      state,
                                      info,
                                      eval,
                                      false_states)
        elif markov_vote:
            next_state, rewards, done, info, steps = self.markov_instantiations[self.markov_idx].run(env,
                                                                   state,
                                                                   info,
                                                                   eval)
            rewards = np.array(rewards)
            termination_vector = self._cumulative_discount_vector[:steps]
            total_reward = np.sum(termination_vector*rewards)
            
            return next_state, total_reward, done, info, steps
        else:
            return None
    
    def _portable_run(self,
                      env,
                      state,
                      info,
                      eval,
                      false_states):
        # run option policy. Takes control from higher-level agent
        steps = 0
        rewards = []
        states = []
        
        logger.info('[option portable run] Starting portable option run')
        
        self.policy.load_buffer(self.policy_buffer_save_file)
        self.policy.move_to_gpu()
        self.termination.move_to_gpu()
        
        with evaluating(self.policy):
            while steps < self.option_timeout:
                if type(state) is np.ndarray:
                    state = torch.from_numpy(state).float()
                states.append(state)
                
                action = self.policy.act(state)
                
                next_state, reward, done, info = env.step(action)
                should_terminate, _, votes, conf = self.termination.vote(next_state)
                steps += 1
                rewards.append(reward)
                
                if steps < self.min_option_length:
                    should_terminate = False
                
                # overwrite reward with option reward
                if should_terminate is True:
                    reward = 1
                else:
                    reward = 0
                self.policy.observe(state, action, reward, next_state, done or should_terminate)
                
                if done or should_terminate:
                    if should_terminate:
                        # option completed successfully
                        logger.info('[portable option run] Option chose to end.')
                        if not eval:
                            self._option_success(states,
                                                next_state,
                                                false_states)
                        termination_vector = self._cumulative_discount_vector[:steps]
                        rewards = np.array(rewards)
                        total_reward = np.sum(termination_vector*rewards)
                        self.policy.store_buffer(self.policy_buffer_save_file)
                        self.policy.move_to_cpu()
                        info["option_timed_out"] = False
                        return next_state, total_reward, done, info, steps
                    else:
                        # episode ended but option did not
                        termination_vector = self._cumulative_discount_vector[:steps]
                        rewards = np.array(rewards)
                        total_reward = np.sum(termination_vector*rewards)
                        self.policy.store_buffer(self.policy_buffer_save_file)
                        self.policy.move_to_cpu()
                        info["option_timed_out"] = False
                        return next_state, total_reward, done, info, steps
                state = next_state
        # option didn't find a valid termination before time limit
        states.append(next_state)
        if not eval:
            self._option_fail(states)
        logger.info('[portable option run] Option timed out')
        termination_vector = self._cumulative_discount_vector[:steps]
        rewards = np.array(rewards)
        total_reward = np.sum(termination_vector*rewards)
        
        self.policy.store_buffer(self.policy_buffer_save_file)
        self.policy.move_to_cpu()
        
        info["option_timed_out"] = True
        
        return next_state, total_reward, done, info, steps
        
    
    def bootstrap_policy(self,
                         bootstrap_envs,
                         max_steps,
                         success_threshold):
        ## initial policy train
        step_number = 0
        steps = []
        episode_number = 0
        total_reward = 0
        rewards = []
        success_queue_size = 50
        # success_queue_size = 500
        success_rates = deque(maxlen=success_queue_size)
        logger.info("[option bootstrap] Bootstrapping option policy...")
        logger.info("[option bootstrap] Success Threshold: {}".format(success_threshold))
        logger.info("[option bootstrap] Max Steps: {}".format(max_steps))
        
        self.policy.move_to_gpu()
        
        env = random.choice(bootstrap_envs)
        
        rand_num = np.random.randint(low=1, high=20)
        state, info = env.reset(agent_reposition_attempts=rand_num)
        
        while step_number < max_steps:
            # action selection
            action = self.policy.act(state)
            
            # step
            next_state, reward, done, info = env.step(action)
            self.policy.observe(state, action, reward, next_state, done)
            total_reward += reward 
            step_number += 1
            state = next_state
            
            if done:
                # check if env sent done signal and agent didn't die
                success_rates.append(reward)
                rewards.append(reward)
                steps.append(step_number)
                well_trained = len(success_rates) == success_queue_size \
                    and np.mean(success_rates) >= success_threshold
                
                if well_trained:
                    logger.info('[option bootstrap] Policy well trained')
                    logger.info('[option bootstrap] Steps: {} Num episodes: {}'.format(step_number, episode_number))
                    logger.info('[option bootstrap] Final Success Rate: {}'.format(np.mean(success_rates)))
                    logger.info('============================================')
                    
                    print('[option bootstrap] Policy well trained')
                    print('[option bootstrap] Steps: {} Num episodes: {}'.format(step_number, episode_number))
                    print('[option bootstrap] Final Success Rate: {}'.format(np.mean(success_rates)))
                    print('============================================')
                    
                    self.policy.store_buffer(self.policy_buffer_save_file)
                    
                    return step_number, total_reward, steps, rewards
                
                episode_number += 1
                env = random.choice(bootstrap_envs)
                
                rand_num = np.random.randint(low=1, high=300)
                state, info = env.reset(agent_reposition_attempts=rand_num)
            
                logger.info('[option bootstrap] {}/{} success rate {}'.format(step_number,
                                                                                int(max_steps),
                                                                                np.mean(success_rates)))
                print('[option bootstrap] {}/{} success rate {}'.format(step_number,
                                                                        int(max_steps),
                                                                        np.mean(success_rates)))
        logger.info('[option bootstrap] Policy did not reach well trained.')
        logger.info('[option bootstrap] Final Success Rate: {}'.format(np.mean(success_rates)))
        
        print('[option bootstrap] Policy did not reach well trained.')
        print('[option bootstrap] Final Success Rate: {}'.format(np.mean(success_rates)))
        
        self.policy.store_buffer(self.policy_buffer_save_file)
        
        return step_number, total_reward, steps, rewards
    
    def initiation_update_confidence(self, was_successful, votes):
        self.initiation.update_confidence(was_successful, votes)
    
    def termination_update_confidence(self, was_successful, votes):
        self.termination.update_confidence(was_successful, votes)

    def _option_success(self,
                        states,
                        termination_state,
                        false_states):
        
        self.create_instance(states,
                             termination_state,
                             false_states)
        
    
    def _option_fail(self,
                     states):
        # option failed so we don't want to create a new instance
        self.initiation.update_confidence(was_successful=False, votes=self.initiation.votes)
        self.initiation.add_data(negative_data=states)
    
    def update_option(self,
                      markov_option,
                      set_epochs):
        logger.info("[update portable option] Updating option with given instance")
        # Update confidences because we now know this was a success
        self.initiation.update_confidence(was_successful=True, votes=markov_option.initiation_votes)
        self.termination.update_confidence(was_successful=True, votes=markov_option.termination_votes)
        # replace replay buffer? maybe should do a random merge or something?
        ########################################
        #       TODO
        #       Figure out what to do with replay buffer
        #       Add data from markov option
        self.policy.load_buffer(self.policy_buffer_save_file)
        
        self.policy = markov_option.policy
        self.initiation.train(set_epochs)
        self.termination.train(set_epochs)
        
        self.save()
