from portable.option.markov import MarkovOption
from portable.option.sets.models import PositionClassifier
from portable.option.policy.agents import evaluating
from contextlib import nullcontext

class PositionMarkovOption(MarkovOption):
    """
    Markov option that uses position to determine the 
    initiation and termination sets
    """
    def __init__(self,
        images,
        positions,
        terminations,
        initial_policy,
        max_option_steps,
        epsilon=2, 
        use_log=True):
        
        super().__init__(use_log)

        self.initiation = PositionClassifier()
        self.initiation.add_positive_examples(images, positions)

        self.policy = initial_policy.initialize_new_policy()
        
        self.termination = terminations
        self.epsilon = epsilon

        self.interaction_count = 0
        self.option_timeout = max_option_steps

    def can_initiate(self, 
                     agent_space_state):
        # Input should be agent's ram position
        return self.initiation.predict(agent_space_state)

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

    def interact_initiation(self, 
                            positive_agent_space_states, 
                            negative_agent_space_states):
        # first element in *_agent_space_states should be image and 
        # second element should be position
        self.initiation.add_positive_examples(positive_agent_space_states[0], positive_agent_space_states[1])
        self.initiation.add_negative_examples(negative_agent_space_states[0], negative_agent_space_states[1])
        self.initiation.fit_classifier()

    def interact_termination(self, 
                             positive_agent_space_states, 
                             negative_agent_space_states):
        
        for state in positive_agent_space_states:
            if state not in self.termination:
                self.termination.append(state)

        for state in negative_agent_space_states:
            if state in self.termination:
                self.termination.remove(state)


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

                should_terminate = self.can_terminate(position)

                self.policy.observe(state, action, reward, next_state, done)
                total_reward += reward

                if done or info['needs_reset'] or should_terminate:
                    # agent died. Should end as failure regardless of if option identified end
                    if done and info['dead']:
                        self.log('[Markov option] Episode ended and we are still executing option. Option failed')
                        info['option_timed_out'] = False
                        positions.append(position)
                        agent_space_states.append(agent_state)
                        self._option_fail({
                            "positions": positions,
                            "agent_space_states":agent_space_states
                        })
                        return next_state, total_reward, done, info, steps
                    # environment needs reset
                    if info['needs_reset']:
                        info['option_timed_out'] = False
                        # option did not fail or succeed. Trajectory won't be used to update option
                        self.log('[Markov option] Environment timed out.')
                        return next_state, total_reward, done, info, steps
                    # option ended 'successfully'
                    if should_terminate:
                        # option completed 'successfully'. 
                        self.log('[Markov option] Option ended "successfully". Ending option')
                        info['option_timed_out'] = False
                        self._option_success({
                            "positions": positions,
                            "agent_space_states":agent_space_states,
                            "termination": position,
                            "agent_space_termination": agent_state
                        })
                        return next_state, total_reward, done, info, steps
                state = next_state

        self.log("[Markov option] Option timed out")
        info["option_timed_out"] = True

        positions.append(position)
        agent_space_states.append(agent_state)

        self._option_fail({
            "positions": positions,
            "agent_space_states":agent_space_states
        })

        return next_state, total_reward, done, info, steps

    def _option_success(self, success_data: dict):
        positions = success_data["positions"]
        agent_space_states = success_data["agent_space_states"]
        termination = success_data["termination"]
        agent_space_termination = success_data["agent_space_termination"]

        self.initiation.add_positive_examples(positions, agent_space_states)
        self.initiation.add_negative_examples([termination], [agent_space_termination])

        self.termination.append(termination)

    def _option_fail(self, failure_data: dict):
        positions = failure_data["positions"]
        agent_space_states = failure_data["agent_space_states"]

        self.initiation.add_negative(
            agent_space_states, positions
        )
    