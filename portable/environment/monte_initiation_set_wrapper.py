import numpy as np
from gym import Wrapper 

from portable.option.sets.models import EnsembleClassifier
from portable.environment.agent_wrapper import build_agent_space_image_stack


class MonteInitiationSetWrapper(Wrapper):
    """
    a wrapper with a hack
    assumes that the agent starts each episode in a place that's not the initiation of the skill,
    and manually drive the agent towards that initiation on reset()

    currently, the manual drive is all just RIGHT actions
    """
    def __init__(self, env, device="cuda"):
        super().__init__(env)
        # load classifier
        clf_path = 'resources/classifier/initiation'
        self.initiation_clf = EnsembleClassifier(device=device)
        self.initiation_clf.load(clf_path)
        # manual action
        self.RIGHT = 3
        assert self.env.unwrapped.get_action_meanings()[self.RIGHT] == 'RIGHT'

    def manually_drive_agent_to_initiation(self):
        """
        the agent started in a place that's not the initiation set of the skill, we manually drive it 
        to the inititation
        
        NOTE: assume the skill is ladder skill, and the initiation can be achieved to keep taking RIGHT actions
        """
        def _is_initiation(env, clf):
            """whether the current time step is initiation"""
            s = build_agent_space_image_stack(env)
            votes, vote_confs = clf.get_votes(s)
            return np.sum(votes) > 0
        # pass through classifier
        termination_counter = 0
        initiate = _is_initiation(self.env, self.initiation_clf)
        while not initiate:
            state, reward, done, info = self.env.step(self.RIGHT)
            initiate = _is_initiation(self.env, self.initiation_clf)
            if done:
                self.env.reset()
                termination_counter += 1
            if termination_counter > 10:
                import matplotlib.pyplot as plt
                image = np.array(state)[-1]
                plt.imsave('results/initiation_failure.png', image)
                raise Exception("failed to drive agent to initiation")
        try:
            return state
        except UnboundLocalError:
            # didn't enter the while loop, `state` is not defined
            return None
    
    def reset(self):
        state = self.env.reset()
        new_state = self.manually_drive_agent_to_initiation()
        if new_state:
            return new_state
        else:
            return state
