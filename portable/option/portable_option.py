import logging
import os 
import numpy as np 
import gin 
import random 
from portable.option.policy.agents import evaluating 
from operator import countOf
from collections import deque

from portable.option.sets import AttentionSet
from portable.option.policy.agents import EnsembleAgent

logger = logging.getLogger(__name__)

@gin.configurable
class AttentionOption():
    def __init__(self,
                 device,
                 log_dir,
                 
                 markov_option_builder,
                 initiation_vote_function,
                 termination_vote_function,
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
                 original_initiation_function,
                 summary_writer=None):
        
        #######################################
        #            TODO                     #
        #######################################
        # add model names
        # summary writer
        
        self.policy = EnsembleAgent(device=device,
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

        self.initiation = AttentionSet(device=device,
                                       vote_function=initiation_vote_function,
                                       beta_distribution_alpha=initiation_beta_distribution_alpha,
                                       beta_distribution_beta=initiation_beta_distribution_beta,
                                       learning_rate=initiation_lr,
                                       dataset_max_size=initiation_dataset_maxsize,
                                       attention_module_num=initiation_attention_module_num,
                                       model_name="initiation",
                                       summary_writer=summary_writer)
        
        self.termination = AttentionSet(device=device,
                                        vote_function=termination_vote_function,
                                        beta_distribution_alpha=termination_beta_distribution_alpha,
                                        beta_distribution_beta=termination_beta_distribution_beta,
                                        learning_rate=termination_lr,
                                        dataset_max_size=termination_dataset_maxsize,
                                        attention_module_num=termination_attention_module_num,
                                        model_name="termination",
                                        summary_writer=summary_writer)

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
        
    @staticmethod
    def _get_save_paths(path):
        policy = os.path.join(path, 'policy')
        initiation = os.path.join(path, 'initiation')
        termination = os.path.join(path, 'termination')
        markov = os.path.join(path, 'markov')

        return policy, initiation, termination, markov

    def save(self, path):
        policy_path, initiation_path, termination_path, markov_path = self._get_save_paths(path)
        
        os.makedirs(policy_path, exist_ok=True)
        os.makedirs(initiation_path, exist_ok=True)
        os.makedirs(termination_path, exist_ok=True)
        
        self.policy.save(policy_path)
        self.initiation.save(initiation_path)
        self.termination.save(termination_path)

        for idx, instance in enumerate(self.markov_instantiations):
            save_path = os.path.join(markov_path, str(idx))
            instance.save(save_path)

        logger.info("[option] Saving option to path: {}".format(path))

    def load(self, path):
        policy_path, initiation_path, termination_path, markov_path = self._get_save_paths(path)

        if os.path.exists(os.path.join(policy_path, 'agent.pkl')):
            self.policy = self.policy.load(os.path.join(policy_path, 'agent.pkl'))
        self.initiation.load(initiation_path)
        self.termination.load(termination_path)

        for idx, instance in self.markov_instantiations:
            save_path = os.path.join(markov_path, str(idx))
            if os.path.exists(save_path):
                instance.load(save_path)

        logger.info("[option] Loading option from path: {}".format(path))

    @staticmethod
    def intersection(list_a, list_b):
        return [i for i in list_a if countOf(list_b,i) > 0]

    @staticmethod
    def disjunctive_union(list_a, list_b):
        return [i for i in list_a if countOf(list_b,i) == 0]
    
    def create_instance(self,
                        states,
                        termination_state):
        ## TODO
        # how to check if we already have instance?
        
        self.markov_instantiations.append(
            self.markov_option_builder(
                states=states,
                termination_state=termination_state,
                initial_policy=self.policy
            )
        )
    
    def can_initiate(self,
                     state):
        logger.info("[portable option initiation] Checking if option can initiate")
        vote_global = self.initiation.vote(state)
        
        local_vote = False
        for idx in range(len(self.markov_instantiations)):
            prediction = self.markov_instantiations[idx].can_initiate(state)
            if prediction == 1:
                self.markov_idx = idx
                local_vote = True
        
        return vote_global, local_vote
    
    def run(self):
        pass 
    
    def bootstrap_policy(self):
        pass
    
    def initiation_update_confidence(self, was_successful, votes):
        self.initiation.update_confidence(was_successful, votes)
    
    def termination_update_confidence(self, was_successful, votes):
        self.termination.update_confidence(was_successful, votes)

    def _option_success(self,
                        states,
                        termination_state):
        termination_votes = self.termination.votes 
        
    
    def _option_fail(self,
                     states):
        # option failed so we don't want to create a new instance
        self.initiation.update_confidence(was_successful=False, votes=self.initiation.votes)
        self.initiation.add_data(negative_data=states)
    
    def update_option(self,
                      markov_option,
                      set_epochs):
        logging.info("[update portable option] Updating option with given instance")
        # Update confidences because we now know this was a success
        self.initiation.update_confidence(was_successful=True, votes=markov_option.initiation_votes)
        self.termination.update_confidence(was_successful=True, votes=markov_option.termination_votes)
        # replace replay buffer? maybe should do a random merge or something?
        ########################################
        #       TODO
        #       Figure out what to do with replay buffer
        #       Add data from markov option
        
        self.policy = markov_option.policy
        self.initiation.train(set_epochs)
        self.termination.train(set_epochs)

