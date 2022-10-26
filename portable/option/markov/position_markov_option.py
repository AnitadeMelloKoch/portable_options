from portable.option.markov import MarkovOption
from portable.option.sets.models import PositionClassifier

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

        while steps < self.option_timeout:
            agent_space_states.append(agent_state)
            positions.append((info["player_x"], info["player_y"]))

            action = self.policy.act(state)

            next_state, reward, done, info = env.step(action)

            agent_state = info["stacked_agent_state"]
            steps += 1

            should_terminate = self.can_terminate((info["player_x"], info["player_y"]))

            self.policy.observe(state, action, reward, next_state, done)
            total_reward += reward

            if done or info['needs_reset'] or should_terminate:
                if (done or should_terminate) and not info['dead']:
                    #TODO
                    pass

                if info['needs_reset']:
                    info['option_timed_out'] = False
                    return next_state, total_reward, done, info, steps

                if done and info['dead']:
                    #TODO
                    pass

            state = next_state

        ## TODO


        return next_state, total_reward, done, info, steps

    