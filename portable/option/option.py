from collections import deque
import logging
import os
import pickle
import numpy as np
import gin
import random
from portable.option.policy.agents import evaluating
from operator import countOf

from portable.option.sets import Set
from portable.option.policy.agents import EnsembleAgent
from portable.utils.utils import plot_state

logger = logging.getLogger(__name__)
import time
import math

import matplotlib.pyplot as plt

@gin.configurable
class Option():
    def __init__(
            self,
            device,

            markov_option_builder,
            get_latent_state,
            initiation_vote_function,
            termination_vote_function,
            policy_phi,
            prioritized_replay_anneal_steps,

            policy_warmup_steps=500000,
            policy_batchsize=32,
            policy_buffer_length=100000,
            policy_update_interval=4,
            q_target_update_interval=40,
            policy_embedding_output_size=64,
            policy_learning_rate=2.5e-4,
            final_epsilon=0.01,
            final_exploration_frames=10**6,
            discount_rate=0.9,
            policy_attention_module_num=8,
            policy_num_output_classes=18,
            stack_size=4,

            initiation_beta_distribution_alpha=30,
            initiation_beta_distribution_beta=5,
            initiation_attention_module_num=8,
            initiation_classifier_learning_rate=1e-4,
            initiation_embedding_learning_rate=1e-2,
            initiation_embedding_output_size=64,
            initiation_dataset_max_size=50000,

            termination_beta_distribution_alpha=30,
            termination_beta_distribution_beta=5,
            termination_attention_module_num=8,
            termination_classifier_learning_rate=1e-4,
            termination_embedding_learning_rate=1e-2,
            termination_embedding_output_size=64,
            termination_dataset_max_size=50000,

            markov_termination_epsilon=3,
            min_interactions=100,
            min_success_rate=0.9,
            timeout=50,
            min_option_length=5,
            original_initiation_function=lambda x:False,

            log=True):
        
        self.policy = EnsembleAgent(
            stack_size=stack_size,
            device=device,
            warmup_steps=policy_warmup_steps,
            batch_size=policy_batchsize,
            phi=policy_phi,
            prioritized_replay_anneal_steps=prioritized_replay_anneal_steps,
            buffer_length=policy_buffer_length,
            update_interval=policy_update_interval,
            q_target_update_interval=q_target_update_interval,
            embedding_output_size=policy_embedding_output_size,
            learning_rate=policy_learning_rate,
            final_epsilon=final_epsilon,
            final_exploration_frames=final_exploration_frames,
            discount_rate=discount_rate,
            num_modules=policy_attention_module_num,
            num_output_classes=policy_num_output_classes
        )

        self.initiation = Set(
            device=device,
            stack_size=stack_size,
            vote_function=initiation_vote_function,
            beta_distribution_alpha=initiation_beta_distribution_alpha,
            beta_distribution_beta=initiation_beta_distribution_beta,
            attention_module_num=initiation_attention_module_num,
            classifier_learning_rate=initiation_classifier_learning_rate,
            embedding_learning_rate=initiation_embedding_learning_rate,
            embedding_output_size=initiation_embedding_output_size,
            dataset_max_size=initiation_dataset_max_size
        )

        self.termination = Set(
            device=device,
            stack_size=stack_size,
            vote_function=termination_vote_function,
            beta_distribution_alpha=termination_beta_distribution_alpha,
            beta_distribution_beta=termination_beta_distribution_beta,
            attention_module_num=termination_attention_module_num,
            classifier_learning_rate=termination_classifier_learning_rate,
            embedding_learning_rate=termination_embedding_learning_rate,
            embedding_output_size=termination_embedding_output_size,
            dataset_max_size=termination_dataset_max_size
        )

        # build markov option so can easily switch
        self.markov_option_builder = markov_option_builder
        # get latent state from environment if needed
        self.get_latent_state = get_latent_state
        self.markov_termination_epsilon = markov_termination_epsilon
        self.markov_instantiations = []
        self.attempted_initiations = []
        self.option_timeout = timeout
        self.markov_min_interactions = min_interactions
        self.markov_min_success_rate = min_success_rate
        self.min_option_length = min_option_length
        self.identify_original_initiation = original_initiation_function

        self.original_markov_initiation = None
        
        self.use_log = log

        
        
    def log(self, message):
        if self.use_log:
            logger.info(message)

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

        self.log("[option] Saving option to path: {}".format(path))

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

        self.log("[option] Loading option from path: {}".format(path))

    @staticmethod
    def intersection(list_a, list_b):
        return [i for i in list_a if countOf(list_b,i) > 0]

    @staticmethod
    def disjunctive_union(list_a, list_b):
        return [i for i in list_a if countOf(list_b,i) == 0]
    
    def create_instance(self,
            images,
            positions,
            terminations,
            termination_images,
            initiation_votes,
            termination_votes
        ):

        common_positions = self.intersection(positions, self.attempted_initiations)

        if len(common_positions) > 0.5*len(positions):
            self.log("[option] instance has been created before. Not creating instance.")
            return
        
        # create new instance of option
        self.markov_instantiations.append(
            self.markov_option_builder(
                images=images,
                positions=positions,
                terminations=terminations,
                termination_images=termination_images,
                initial_policy=self.policy,
                max_option_steps=self.option_timeout,
                initiation_votes=initiation_votes,
                termination_votes=termination_votes,
                min_required_interactions=self.markov_min_interactions,
                success_rate_required=self.markov_min_success_rate,
                assimilation_min_required_interactions=self.markov_min_interactions//2,
                assimilation_success_rate_required=self.markov_min_success_rate,
                epsilon=self.markov_termination_epsilon,
                use_log=self.use_log
            )
        )
        
        self.attempted_initiations += self.disjunctive_union(positions, self.attempted_initiations)

        self.log("[option] New instantiation created")

    def can_initiate(self, state, env, info):
        # check if option can initiate
        latent_state = self.get_latent_state(env, info)
        self.log("[option] Checking initiation")
        if latent_state in self.attempted_initiations:
            vote_global = 0
        else:
            vote_global = self.initiation.vote(state)
        vote_markov = False
        self.markov_idx = None
        for idx in range(len(self.markov_instantiations)):
            prediction = self.markov_instantiations[idx].can_initiate(latent_state)
            if prediction == 1:
                self.markov_idx = idx
                vote_markov = True
        
        return vote_global, vote_markov

    def run(self, 
            env, 
            state, 
            info,
            eval):
        
        # latent_state = (info["player_x"], info["player_y"], info["has_key"], info["door_open"])
        latent_state = self.get_latent_state(env, state)
        global_vote, markov_vote = self.can_initiate(state, latent_state)

        if global_vote and not markov_vote:
            if eval is False:
                return self._portable_run(env, state, info)
            else:
                return self._portable_evaluate(env, state, info)
        elif markov_vote:
            return self.markov_instantiations[self.markov_idx].run(
                env, state, info, eval
            )
        else:
            return None

    def _portable_run(self, 
            env, 
            state, 
            info):
        # run the option. Policy takes over control
        steps = 0
        total_reward = 0
        states = []
        latent_states = []
        latent_state = self.get_latent_state(env, state)

        self.log("[portable option:run] Begining run for portable option")

        while steps < self.option_timeout:
            states.append(state)
            latent_states.append(latent_state)
            
            action = self.policy.act(state)

            next_state, reward, done, info = env.step(action)
            state = next_state
            latent_state = self.get_latent_state(env, state)
            steps += 1
            total_reward += reward

            self.log("[portable option:run] checking termination")
            should_terminate = self.termination.vote(next_state)
            # going to limit how long option must run before
            # we let it finish maybe shouldn't do this?
            if steps < self.min_option_length:
                should_terminate = False
            
            # overwrite reward with reward for option
            if should_terminate is True:
                reward = 1
            else:
                reward = 0
            self.policy.observe(state, action, reward, next_state, done or should_terminate, update_policy=False)

            # need death condition => maybe just terminate on life lost wrapper
            if done or info['needs_reset'] or should_terminate:
                # environment needs reset (not time limited task)
                if info['needs_reset']:
                    self.log('[portable option:run] Environment timed out')
                    info['option_timed_out'] = False
                    return next_state, total_reward, done, info, steps
                if should_terminate:
                    # option completed successfully
                    self.log('[portable option:run] Option ended successfully. Ending option')
                    info['option_timed_out'] = False
                    self._option_success(
                        latent_states,
                        states,
                        latent_state,
                        next_state
                    )
                    return next_state, total_reward, done, info, steps
                
            state = next_state

        # we didn't find a valid termination for the option before the 
        # allowed execution time ran out => fail
        latent_states.append(latent_state)
        states.append(next_state)
        self._option_fail(states)
        self.log("[option] Option timed out. Returning\n\t {}".format(info['position']))
        info['option_timed_out'] = True

        return next_state, total_reward, done, info, steps
            
    def _portable_evaluate(self,
                env,
                state,
                info):
        # runs option in evaluation mode (does not store transitions for later training)
        # does not create Markov instantiations
        steps = 0
        total_reward = 0
        with evaluating(self.policy):
            while steps < self.option_timeout:
                action = self.policy.act(state)

                next_state, reward, done, info = env.step(action)
                steps += 1
                # environment reward for use outside option
                total_reward += reward

                self.log("[portable option:eval] checking termination")
                should_terminate = self.termination.vote(next_state)
                if steps < self.min_option_length:
                    should_terminate = False

                # get option reward for policy
                reward = 1 if should_terminate else 0

                self.policy.observe(state, action, reward, next_state, done or should_terminate)

                if done or info['needs_reset'] or should_terminate:
                    if (done or should_terminate):
                        self.log("[portable option:eval] Option completed successfully (done: {}, should_terminate {})".format(done, should_terminate))
                        info['option_timed_out'] = False
                        return next_state, total_reward, done, info, steps
                        
                    if info['needs_reset']:
                        self.log("[portable option:eval] Environment needs reset. Returning")
                        info['option_timed_out'] = False
                        return next_state, total_reward, done, info, steps
                
                state = next_state

        self.log("[portable option:eval] Option timed out. Returning")
        info['option_timed_out'] = True

        return next_state, total_reward, done, info, steps

    def bootstrap_policy(self, 
                         bootstrap_envs,
                         max_steps,
                         success_rate_for_well_trained):
        # initial policy train
        # bootstrap_env: the environment the agent policy is trained on
        #       this environment should reset to an appropriate location and
        #       end episode (with appropriate reward) when the option policy has 
        #       hit a termination
        # max_steps: maximum number of steps the policy can take to train
        step_number = 0
        steps = []
        episode_number = 0
        total_reward = 0
        rewards = []
        success_queue_size = 500
        success_rates = deque(maxlen=success_queue_size)
        self.log("[option] Bootstrapping option policy...")
        
        env = random.choice(bootstrap_envs)

        state, info = env.reset()

        while step_number < max_steps:
            # action selection
            action = self.policy.act(state)

            # step
            next_state, reward, done, info = env.step(action)
            #  really need to change this too
            self.policy.observe(state, action, reward, next_state, done)
            total_reward += reward
            step_number += 1
            state = next_state

            if done or info['needs_reset']:
                # check if env sent done signal and agent didn't die
                success_rates.append(reward)
                rewards.append(reward)
                steps.append(step_number)
                well_trained = len(success_rates) == success_queue_size \
                    and np.mean(success_rates) >= success_rate_for_well_trained

                if well_trained:
                    self.log('[option bootstrap] Policy well trained in {} steps and {} episodes. Success rate {}'.format(step_number, episode_number, np.mean(success_rates)))
                    print('[option] Policy well trained in {} steps and {} episodes. Success rate {}'.format(step_number, episode_number, np.mean(success_rates)))
                    return step_number, total_reward, steps, rewards

                episode_number += 1
                env = random.choice(bootstrap_envs)

                state, info = env.reset()
        
                pos = (info["player_x"], 
                        info["player_y"], 
                        info["has_key"], 
                        info["door_open"],
                        info["seed"])
                if pos not in self.attempted_initiations:
                    self.attempted_initiations.append(pos)
        
                if (episode_number - 1) % 50 == 0:
                    self.log("[option bootstrap] Completed Episode {} steps {} success rate {}".format(episode_number-1, step_number, np.mean(success_rates)))
                    print("[option] Completed Episode {} steps {} success rate {}".format(episode_number-1, step_number, np.mean(success_rates)))

        self.log("[option] Policy did not reach well trained threshold. Success rate {}".format(np.mean(success_rates)))
        print("[option] Policy did not reach well trained threshold. Success rate {}".format(np.mean(success_rates)))
        return step_number, total_reward, steps, rewards


    def initiation_update_confidence(self, was_successful, votes):
        self.initiation.update_confidence(was_successful, votes)

    def termination_update_confidence(self, was_successful, votes):
        self.termination.update_confidence(was_successful, votes)

    def _option_success(
            self,
            positions,
            states,
            markov_termination,
            termination):
        
        # we successfully completed the option once so we want to create a new instance
        # assume an existing instance does not exist because we attempted to use the portable option

        # should do this better, need to change the sets maybe or maybe this is fine
        termination_votes = self.termination.votes
        # not sure this is right
        initiation_votes = self.initiation.votes

        # really need to check that these inputs are correct because I don't think they are :/
        self.create_instance(
            states,
            positions,
            [markov_termination],
            [termination],
            initiation_votes,
            termination_votes
        )

    def _option_fail(self,
                     states):
        # option failed so do not create a new instanmce and downvote initiation sets that triggered
        self.initiation.update_confidence(was_successful=False, votes=self.initiation.votes)
        self.initiation.add_data(negative_data=states)
        
    def update_option(self, 
                      markov_option,
                      policy_epochs,
                      embedding_epochs,
                      classifier_epochs):
        logger.info("[portable option:update_option] Updating option with given instance")
        # Update confidences because we now know this was a success
        self.initiation.update_confidence(was_successful=True, votes=markov_option.initiation_votes)
        self.termination.update_confidence(was_successful=True, votes=markov_option.termination_votes)
        # Just going to replace replay buffer which contains the new samples
        self.policy.replay_buffer = markov_option.policy.replay_buffer
        # Add new samples from instance to classifier datasets
        positive_samples_init, negative_samples_init = markov_option.initiation.get_images()
        self.initiation.add_data(positive_data=positive_samples_init, negative_data=negative_samples_init)
        self.termination.add_data(
            positive_data = markov_option.termination_images,
            negative_data=random.sample(positive_samples_init, len(markov_option.termination_images))
        )

        # train policy and classifiers
        # self.policy.train(policy_epochs)
        self.policy = markov_option.policy
        self.initiation.train(
            embedding_epochs,
            classifier_epochs
        )
        self.termination.train(
            embedding_epochs,
            classifier_epochs
        )