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
        

class PerfectDoorOpen(BaseTermination):
    def __init__(self, door_colour=None):
        super().__init__()
        self.door_colour = door_colour
        self.x = None
        self.y = None
    
    def get_door(self, env):
        if self.x and self.y:
            cell = env.unwrapped.grid.get(self.x, self.y)
            if cell:
                if cell.type == "door":
                    return cell
        
        for x in range(env.unwrapped.width):
            for y in range(env.unwrapped.height):
                cell = env.unwrapped.grid.get(x,y)
                if cell:
                    if cell.type == "door":
                        self.x = x
                        self.y = y
                        return cell
    
    def check_term(self, state, env):
        door = self.get_door(env)
        if self.door_colour:
            return door.is_open and (door.color == self.door_colour)
        return door.is_open

class NeverPerfectDoorOpen(BaseTermination):
    def __init__(self, door_colour=None):
        super().__init__()
        self.door_colour = door_colour
        self.x = None
        self.y = None
    
    def get_door(self, env):
        if self.x and self.y:
            cell = env.unwrapped.grid.get(self.x, self.y)
            if cell:
                if cell.type == "door":
                    return cell
        
        for x in range(env.unwrapped.width):
            for y in range(env.unwrapped.height):
                cell = env.unwrapped.grid.get(x,y)
                if cell:
                    if cell.type == "door":
                        self.x = x
                        self.y = y
                        return cell
    
    def check_term(self, state, env):
        door = self.get_door(env)
        if self.door_colour:
            return not (door.is_open and (door.color == self.door_colour))
        return not door.is_open

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

class PerfectAtLocation(BaseTermination):
    def __init__(self,
                 x,
                 y):
        super().__init__()
        self.x = x 
        self.y = y
    
    def check_term(self, state, env):
        agent_x, agent_y = env.unwrapped.agent_pos
        return (agent_x == self.x and agent_y == self.y)

class RandomAtLocation(BaseTermination):
    def __init__(self, x, y, p):
        super().__init__()
        self.x = x
        self.y = y
        self.p = p
    
    def check_term(self, state, env):
        if state in self.seen_states.keys():
            return self.seen_states[state]
        else:
            term = None
            if np.random.rand() < self.p:
                agent_x, agent_y = env.unwrapped.agent_pos
                term = (agent_x == self.x and agent_y == self.y)
            else:
                term = np.random.rand() < 0.5
            
            self.seen_states[state] = term
            return term
