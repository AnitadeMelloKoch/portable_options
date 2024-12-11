from portable.option.markov import MarkovOption
from portable.option.sets.models import PositionClassifier
from portable.option.policy.agents import evaluating
from contextlib import nullcontext
import os
import pickle
import numpy as np
from collections import deque


class PositionMarkovOption(MarkovOption):
    """
    Markov option that uses position to determine the 
    initiation and termination sets
    """
    def __init__(self,
        images,
        positions,
        terminations,
        termination_images,
        initial_policy,
        max_option_steps,
        initiation_votes,
        termination_votes,
        min_required_interactions,
        success_rate_required,
        assimilation_min_required_interactions,
        assimilation_success_rate_required,
        save_file,
        epsilon=2, 
        use_log=True):
        
        super().__init__(use_log)

        self.save_file = save_file
        self.initiation = PositionClassifier()
        self.initiation.add_positive_examples(images, positions)
        self.initiation.fit_classifier()

        self.policy = initial_policy.initialize_new_policy()
        self.initiation_votes = initiation_votes
        self.termination_votes = termination_votes
        
        self.termination = terminations
        self.termination_images = termination_images
        self.epsilon = epsilon

        self.interaction_count = 0
        self.option_timeout = max_option_steps

        self.performance = deque(maxlen=min_required_interactions)
        self.min_interactions = min_required_interactions
        self.success_rate_required = success_rate_required
        self.assimilation_performance = deque(maxlen=assimilation_min_required_interactions)
        self.assimilation_min_interactions = assimilation_min_required_interactions
        self.assimilation_success_rate_required = assimilation_success_rate_required
        
        self.policy.store_buffer(save_file=self.save_file)

    @staticmethod
    def _get_save_paths(path):
        policy = os.path.join(path, 'policy')
        initiation = os.path.join(path, 'initiation')
        termination = os.path.join(path, 'termination')

        return policy, initiation, termination

    def save(self, path: str):
        policy_path, initiation_path, termination_path = self._get_save_paths(path)
        
        os.makedirs(policy_path, exist_ok=True)
        os.makedirs(initiation_path, exist_ok=True)
        os.makedirs(termination_path, exist_ok=True)
        
        self.policy.save(policy_path)
        self.initiation.save(initiation_path)

        with open(os.path.join(termination_path, 'terminations.pkl'), "wb") as f:
            pickle.dump(self.termination, f)
        
        np.save(os.path.join(termination_path, 'epsilon.npy'), self.epsilon)

    def load(self, path: str):
        policy_path, initiation_path, termination_path = self._get_save_paths(path)
        
        self.policy.load(policy_path)
        self.initiation.load(initiation_path)

        if os.path.exists(os.path.join(termination_path, 'terminations.pkl')):
            with open(termination_path, 'terminations.pkl', "rb") as f:
                self.termination = pickle.load(f)

        if os.path.exists(os.path.join(termination_path, 'epsilon.npy')):
            self.epsilon = np.load(os.path.join(termination_path, 'epsilon.npy'))

    def can_initiate(self, 
                     state,
                     info):
        # Input should be agent's ram position
        position = [info["position"][0], info["position"][1]]
        return self.initiation.predict(position)

    def can_terminate(self, 
                      agent_space_state):
        
        def in_epsilon_square(pos, center):
            if pos[0] <= (center[0]+self.epsilon) and \
                pos[0] >= (center[0]-self.epsilon) and \
                pos[1] <= (center[1]+self.epsilon) and \
                pos[1] >= (center[1]-self.epsilon):
                return True
            return False

        for state in self.termination:
            if in_epsilon_square(agent_space_state, state):
                return True
        return False


    def run(self, 
            env, 
            state, 
            info, 
            evaluate):
            
        steps = 0
        total_reward = 0
        agent_space_states = []
        agent_state = info["stacked_agent_state"]
        positions = []
        position = (info["player_x"], info["player_y"])
        
        self.policy.load_buffer(save_file=self.save_file)
        self.policy.move_to_gpu()

        # if we are evaluating then set policy to eval mode else ignore
        with evaluating(self.policy) if evaluate else nullcontext():
            while steps < self.option_timeout:
                agent_space_states.append(agent_state)
                positions.append(position)

                action = self.policy.act(state)

                next_state, reward, done, info = env.step(action)

                agent_state = info["stacked_agent_state"]
                position = (info["player_x"], info["player_y"])
                steps += 1
                # we are storing the environment reward to be used outside the agent
                total_reward += reward

                should_terminate = self.can_terminate(position)

                # overwrite reward with reward for option
                if should_terminate:
                    reward = 1
                else:
                    reward = 0

                self.policy.observe(state, action, reward, next_state, done or should_terminate)

                if done or info['needs_reset'] or should_terminate:
                    # agent died. Should end as failure regardless of if option identified end
                    if info['dead']:
                        self.log('[Markov option] Agent died. Option failed')
                        info['option_timed_out'] = False
                        positions.append(position)
                        agent_space_states.append(agent_state)
                        if evaluate:
                            self._option_fail({
                                "positions": positions,
                                "agent_space_states":agent_space_states
                            })
                        self.policy.store_buffer(save_file=self.save_file)
                        self.policy.move_to_cpu()
                        return next_state, total_reward, done, info, steps
                    if done and not should_terminate:
                        self.log('[Markov option] Episode ended and we are still executing option. Option failed')
                        info['option_timed_out'] = False
                        positions.append(position)
                        agent_space_states.append(agent_state)
                        if evaluate:
                            self._option_fail({
                                "positions": positions,
                                "agent_space_states":agent_space_states
                            })
                        self.policy.store_buffer(save_file=self.save_file)
                        self.policy.move_to_cpu()
                        return next_state, total_reward, done, info, steps
                    # environment needs reset
                    if info['needs_reset']:
                        info['option_timed_out'] = False
                        # option did not fail or succeed. Trajectory won't be used to update option
                        self.log('[Markov option] Environment timed out.')
                        self.policy.store_buffer(save_file=self.save_file)
                        self.policy.move_to_cpu()
                        return next_state, total_reward, done, info, steps
                    # option ended 'successfully'
                    if should_terminate:
                        # option completed 'successfully'. 
                        self.log('[Markov option] Option ended "successfully". Ending option')
                        info['option_timed_out'] = False
                        if evaluate:
                            self._option_success({
                                "positions": positions,
                                "agent_space_states":agent_space_states,
                                "termination": position,
                                "agent_space_termination": agent_state
                            })
                        self.policy.store_buffer(save_file=self.save_file)
                        self.policy.move_to_cpu()
                        return next_state, total_reward, done, info, steps
                state = next_state

        self.log("[Markov option] Option timed out")
        info["option_timed_out"] = True

        positions.append(position)
        agent_space_states.append(agent_state)
        
        
        if evaluate:
            self._option_fail({
                "positions": positions,
                "agent_space_states":agent_space_states
            })

        self.policy.store_buffer(save_file=self.save_file)
        self.policy.move_to_cpu()
        
        return next_state, total_reward, done, info, steps

    def can_assimilate(self):
        if len(self.assimilation_performance) < self.assimilation_min_interactions:
            return None
        if np.mean(self.assimilation_performance) >= self.assimilation_success_rate_required:
            return True
        else:
            return False

    def is_well_trained(self):
        if len(self.performance) < self.min_interactions:
            return False
        if np.mean(self.performance) >= self.success_rate_required:
            return True
        else:
            return False

    def assimilate_run(self,
                       env,
                       state,
                       info):
        steps = 0
        total_reward = 0
        position = (info["player_x"], info["player_y"])
        
        self.policy.load_buffer(save_file=self.save_file)
        self.policy.move_to_gpu()

        with evaluating(self.policy):
            while steps < self.option_timeout:

                action = self.policy.act(state)

                next_state, reward, done, info = env.step(action)

                position = (info["player_x"], info["player_y"])
                steps += 1
                total_reward += reward

                should_terminate = self.can_terminate(position)

                if should_terminate:
                    reward = 1
                else:
                    reward = 0

                self.policy.observe(state, action, reward, next_state, done or should_terminate)

                if done or info['needs_reset'] or should_terminate:
                    if info["dead"]:
                        self.log('[assimilation test] Agent died. Option failed.')
                        info['option_timed_out'] = False
                        self.assimilation_performance.append(0)
                        
                        self.policy.store_buffer(save_file=self.save_file)
                        self.policy.move_to_cpu()
                        
                        return next_state, total_reward, done, info, steps

                    if done and not should_terminate:
                        self.log('[assimilation test] Episode ended but we are still executing option. Option failed.')
                        info['option_timed_out'] = False
                        self.assimilation_performance.append(0)
                        
                        self.policy.store_buffer(save_file=self.save_file)
                        self.policy.move_to_cpu()
                        
                        return next_state, total_reward, done, info
                    
                    if info['needs_reset']:
                        info['option_timed_out'] = False
                        self.log('[assimilation test] Environment timed out.')
                        
                        self.policy.store_buffer(save_file=self.save_file)
                        self.policy.move_to_cpu()
                        
                        return next_state, total_reward, done, info, steps
                    if should_terminate:
                        self.log('[assimilation test] Option ended successfully. Ending option')
                        info['option_timed_out'] = False
                        self.assimilation_performance.append(1)
                        
                        self.policy.store_buffer(save_file=self.save_file)
                        self.policy.move_to_cpu()
                        
                        return next_state, total_reward, done, info, steps
            state = next_state
        self.log("[assimilation test] Option timed out")
        info["option_timed_out"] = True
        self.assimilation_performance.append(0)
        
        self.policy.store_buffer(save_file=self.save_file)
        self.policy.move_to_cpu()
        
        return next_state, total_reward, done, info, steps

    def _option_success(self, success_data: dict):
        positions = success_data["positions"]
        agent_space_states = success_data["agent_space_states"]
        termination = success_data["termination"]
        agent_space_termination = success_data["agent_space_termination"]

        self.initiation.add_positive_examples(agent_space_states,positions)
        self.initiation.add_negative_examples([agent_space_termination],[termination])
        self.initiation.fit_classifier()

        self.termination.append(termination)
        self.performance.append(1)

        self.log('[Markov option] Success Rate: {} Num interactions: {}'.format(np.mean(self.performance), len(self.performance)))


    def _option_fail(self, failure_data: dict):
        positions = failure_data["positions"]
        agent_space_states = failure_data["agent_space_states"]

        self.initiation.add_negative_examples(
            agent_space_states, positions
        )
        self.initiation.fit_classifier()
        self.performance.append(0)
    
        self.log('[Markov option] Success Rate: {} Num interactions: {}'.format(np.mean(self.performance), len(self.performance)))
