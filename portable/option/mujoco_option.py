import logging
import numpy as np
import gin
import torch.nn as nn
from torch import distributions
import torch
from pfrl.nn.lmbda import Lambda
import pfrl

from portable.option.sets import Set
from portable.option.sets.utils import PositionSetPair
from portable.option.policy.agents import SAC
from portable.option import Option
from portable.option.policy.agents.abstract_agent import evaluating

logger = logging.getLogger(__name__)



@gin.configurable
class MujocoOption(Option):
    def __init__(
            self, 
            device, 
            initiation_vote_function, 
            termination_vote_function, 
            policy_observation_shape,
            env_action_space,
            policy_hidden_channels=1024,
            policy_n_step_return=3,
            policy_gamma=0.98,
            policy_replay_start_size=10000,
            policy_batchsize=32, 
            policy_buffer_length=100000, 
            policy_update_interval=4, 
            policy_learning_rate=0.00025, 
            initiation_beta_distribution_alpha=30, 
            initiation_beta_distribution_beta=5, 
            initiation_attention_module_num=8, 
            initiation_embedding_learning_rate=0.0001, 
            initiation_classifier_learning_rate=0.01, 
            initiation_embedding_output_size=64, 
            initiation_dataset_max_size=50000, 
            termination_beta_distribution_alpha=30, 
            termination_beta_distribution_beta=5, 
            termination_attention_module_num=8, 
            termination_embedding_learning_rate=0.0001, 
            termination_classifier_learning_rate=0.01, 
            termination_embedding_output_size=64, 
            termination_dataset_max_size=50000, 
            markov_termination_epsilon=3, 
            min_interactions=100, 
            q_variance_threshold=1, 
            timeout=50, 
            allowed_additional_loss=2, 
            log=True):
        super().__init__(device, 
                         initiation_vote_function, 
                         termination_vote_function, 
                         lambda x : 0, 
                         "ucb_leader", 
                         1000, 
                         0, 
                         32, 
                         64,
                         4, 
                         10, 
                         64, 
                         1, 
                         1, 
                         10, 
                         0.1, 
                         1, 
                         4, 
                         initiation_beta_distribution_alpha, 
                         initiation_beta_distribution_beta, 
                         initiation_attention_module_num, 
                         initiation_embedding_learning_rate, 
                         initiation_classifier_learning_rate, 
                         initiation_embedding_output_size, 
                         initiation_dataset_max_size, 
                         termination_beta_distribution_alpha, 
                         termination_beta_distribution_beta, 
                         termination_attention_module_num, 
                         termination_embedding_learning_rate, 
                         termination_classifier_learning_rate, 
                         termination_embedding_output_size, 
                         termination_dataset_max_size, 
                         markov_termination_epsilon, 
                         min_interactions, 
                         q_variance_threshold, 
                         timeout, 
                         allowed_additional_loss, 
                         log)

        action_size = env_action_space.shape[0]
        def squashed_diagonal_gaussian_head(x):
                assert x.shape[-1] == action_size * 2
                mean, log_scale = torch.chunk(x, 2, dim=1)
                log_scale = torch.clamp(log_scale, -20.0, 2.0)
                var = torch.exp(log_scale * 2)
                base_distribution = distributions.Independent(
                    distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1
                )
                # cache_size=1 is required for numerical stability
                return distributions.transformed_distribution.TransformedDistribution(
                    base_distribution, [distributions.transforms.TanhTransform(cache_size=1)]
                )
        policy = nn.Sequential(
            nn.Linear(policy_observation_shape, policy_hidden_channels),
            nn.ReLU(),
            nn.Linear(policy_hidden_channels, policy_hidden_channels),
            nn.ReLU(),
            nn.Linear(policy_hidden_channels, action_size*2),
            Lambda(squashed_diagonal_gaussian_head)
        )
        torch.nn.init.xavier_uniform_(policy[0].weight)
        torch.nn.init.xavier_uniform_(policy[2].weight)
        torch.nn.init.xavier_uniform_(policy[4].weight)
        policy_optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=policy_learning_rate
        )
        def make_q_func_with_optim():
            q_func = nn.Sequential(
                pfrl.nn.ConcatObsAndAction(),
                nn.Linear(policy_observation_shape+action_size, policy_hidden_channels),
                nn.ReLU(),
                nn.Linear(policy_hidden_channels, policy_hidden_channels),
                nn.ReLU(),
                nn.Linear(policy_hidden_channels, 1)
            )
            torch.nn.init.xavier_uniform_(q_func[1].weight)
            torch.nn.init.xavier_uniform_(q_func[3].weight)
            torch.nn.init.xavier_uniform_(q_func[5].weight)
            q_func_optimizer = torch.optim.Adam(
                q_func.parameters(), lr=policy_learning_rate
            )
            return q_func, q_func_optimizer
        q_func1, q_func1_optimizer = make_q_func_with_optim()
        q_func2, q_func2_optimizer = make_q_func_with_optim()

        rbuf = pfrl.replay_buffers.ReplayBuffer(policy_buffer_length, policy_n_step_return)

        def burnin_action_func():
            """Select random actions until model is updated one or more times."""
            return env_action_space.sample()

        self.policy = SAC(
            policy,
            q_func1,
            q_func2,
            policy_optimizer,
            q_func1_optimizer,
            q_func2_optimizer,
            rbuf,
            policy_gamma,
            policy_update_interval,
            policy_replay_start_size,
            gpu=0,
            minibatch_size=policy_batchsize,
            burnin_action_func=burnin_action_func,
            entropy_target=-action_size,
            temperature_optimizer_lr=policy_learning_rate
        )

    def one_step(self, env, state, steps, env_max_steps):
        # execute one step
        action = self.policy.batch_act(state)
        next_state, reward, done, infos = env.step(action)
        steps += 1
        reset = steps == env_max_steps
        steps[done] = 0

        self.policy.batch_observe(
            batch_obs = next_state,
            batch_reward = reward,
            batch_done = done,
            batch_reset = reset
        )

        # get step statistics
        epinfo = []
        for inf in infos:
            maybe_epinfo = inf.get('episode')
            if maybe_epinfo:
                epinfo.append(maybe_epinfo)

        return next_state, reward, done, steps, infos, epinfo

    def run(self, 
            env, 
            state, 
            info, 
            steps,
            q_values, 
            option_q, 
            use_global_classifiers=True):

        # run the option. Policy takes over control
        option_steps = 0
        total_reward = 0
        agent_space_states = []
        agent_state = env.render_camera(imshow=False)
        positions = []
        positions.append(info[0]["position"])

        while option_steps < self.option_timeout:
            agent_space_states.append(agent_state)
            
            next_state, reward, done, steps, infos, epinfo = self.one_step(env, state, steps, self.option_timeout*100)
            # next_state, reward, done, info = env.step(action)
            # agent_state = info["stacked_agent_state"]
            position = infos[0]["position"]
            positions.append(position)
            agent_state = env.imshow(imshow=False)
            option_steps += 1

            if use_global_classifiers:
                should_terminate = self.termination.vote(agent_state)
            else:
                should_terminate = self.markov_classifiers[self.markov_classifier_idx].can_terminate(position)
            
            total_reward += reward

            if done or should_terminate:
                # ended episode without dying => success
                # detected the end of the option => success
                if should_terminate:
                    self._option_success(
                        positions,
                        agent_space_states,
                        info,
                        agent_state,
                        q_values,
                        option_q
                    )
                    self.log("[option run] Option completed successfully (done: {}, should_terminate: {})".format(done, should_terminate))
                    infos[0]['option_timed_out'] = False
                    return next_state, total_reward, done, infos, steps

                # we died during option execution => fail
                if done and not should_terminate:
                    self.log("[option run] Option failed. Returning \n\t {}".format(position))
                    self._option_fail(
                        positions,
                        agent_space_states
                    )
                    infos[0]['option_timed_out'] = False
                    return next_state, total_reward, done, infos, steps
            
            state = next_state

        # we didn't find a valid termination for the option before the 
        # allowed execution time ran out => fail
        self._option_fail(
            positions,
            agent_space_states
        )
        self.log("[option] Option timed out. Returning\n\t {}".format(position))
        infos[0]['option_timed_out'] = True

        return next_state, total_reward, done, infos, steps

    def evaluate(self,
                 env,
                 state,
                 info,
                 steps,
                 use_global_classifiers=True):
        # run the option. Policy takes over control
        option_steps = 0
        total_reward = 0
        agent_state = env.render_camera(imshow=False)

        with evaluating(self.policy):
            while option_steps < self.option_timeout:
                
                next_state, reward, done, infos, epinfo = self.one_step(env, state, steps, info)
                # next_state, reward, done, info = env.step(action)
                # agent_state = info["stacked_agent_state"]
                position = infos[0]["position"]
                agent_state = env.imshow(imshow=False)
                option_steps += 1

                if use_global_classifiers:
                    should_terminate = self.termination.vote(agent_state)
                else:
                    should_terminate = self.markov_classifiers[self.markov_classifier_idx].can_terminate(position)
                
                total_reward += reward

                if done or should_terminate:
                    # ended episode without dying => success
                    # detected the end of the option => success
                    if should_terminate:
                        self.log("[option run] Option completed successfully (done: {}, should_terminate: {})".format(done, should_terminate))
                        infos[0]['option_timed_out'] = False
                        return next_state, total_reward, done, infos, steps

                    # we died during option execution => fail
                    if done and not should_terminate:
                        self.log("[option run] Option failed. Returning \n\t {}".format(position))
                        infos[0]['option_timed_out'] = False
                        return next_state, total_reward, done, infos, steps
                
                state = next_state

        # we didn't find a valid termination for the option before the 
        # allowed execution time ran out => fail
        self.log("[option] Option timed out. Returning\n\t {}".format(position))
        infos[0]['option_timed_out'] = True

        return next_state, total_reward, done, infos, steps