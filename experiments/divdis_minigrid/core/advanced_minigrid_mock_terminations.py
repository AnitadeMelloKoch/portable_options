# set of hand-defined termination classifers for debugging meta policy
# methods
import numpy as np

def check_got_key(env, colour):
    if env.unwrapped.carrying is not None:
        if any(obj.type=='key' for obj in env.unwrapped.carrying):
            return any((obj.type=='key' and obj.color==colour) for obj in env.unwrapped.carrying)
    return False

def wrong_check_got_key(env, colour):
    if env.unwrapped.carrying is not None:
        if any(obj.type=='key' for obj in env.unwrapped.carrying):
            return all((obj.type=='key' and obj.color!=colour) for obj in env.unwrapped.carrying)
    return True

class BaseTermination():
    def __init__(self):
        self.seen_states = {}
    
    def check_term(self, state, env):
        return False
    
    def __call__(self, state, env):
        return self.check_term(state, env)
        

class PerfectGetKey(BaseTermination):
    def __init__(self, key_colour):
        self.key_colour = key_colour
        super().__init__()
    
    def check_term(self, state, env):
        # check got red key
        return check_got_key(env, self.key_colour)

class NeverCorrectGetKey(BaseTermination):
    def __init__(self, key_colour):
        self.key_colour = key_colour
        super().__init__()
    
    def check_term(self, state, env):
        return wrong_check_got_key(env, self.key_colour)

class ProbabilisticGetKey(BaseTermination):
    def __init__(self, 
                 key_colour,
                 prob_correct):
        self.key_colour = key_colour
        self.prob_correct = prob_correct
        super().__init__()
    
    def check_term(self, state, env):
        if state in self.seen_states.keys():
            return self.seen_states[state]
        else:
            randval = np.random.rand()
            if randval <= self.prob_correct:
                term = check_got_key(env, self.key_colour)
            else:
                term = wrong_check_got_key(env, self.key_colour)
            
            self.seen_states[state] = term
            return term

class RandomGetKey(BaseTermination):
    def __init__(self):
        super().__init__()
    
    def check_term(self, state, env):
        if state in self.seen_states.keys():
            return self.seen_states[state]
        else:
            term = None
            randval = np.random.rand()
            if randval < 0.5:
                term = True
            else:
                term = False
            
            self.seen_states[state] = term
            return term

