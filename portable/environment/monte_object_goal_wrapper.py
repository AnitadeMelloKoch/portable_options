from gym import Wrapper

from portable.utils import get_player_room_number


class MonteObjectGoalWrapper(Wrapper):
    """
    This is the base class for all wrappers that deal with monte moving around some object as a goal
    
    This class is not intended to be used directly, but instead contain common code for specific wrappers
    Currently used by the following wrappers:
        - MonteSkullGoalWrapper
        - MonteSpiderGoalWrapper
        - MonteSnakeGoalWrapper
    """
    def __init__(self, env, epsilon_tol=8, info_only=False):
        """
        Args:
            epsilon_tol: how close to the object the agent must be to finish the skill
            info_only: if True, don't override the reward and done from the environment, 
                        but only add a field `reached_goal` to the info dict
        """
        super().__init__(env)
        self.env = env
        self.epsilon_tol = epsilon_tol
        self.info_only = info_only
        self.room_number = get_player_room_number(self.env.unwrapped.ale.getRAM())
    
    def reached_goal(self, player_x, player_y, object_x, room_number):
        """
        determine if the player has reached the goal
        """
        on_ground = player_y == self.y
        to_the_left = player_x < object_x and abs(player_x - object_x) < self.epsilon_tol
        in_same_room = room_number == self.room_number
        return on_ground and to_the_left and in_same_room
    
    def finished_skill(self, player_x, player_y, object_x, room_number, done, info):
        """
        determine if the monte agent has finished the skill
        The agent finishes the skill if the player is:
            - to the left of the object
            - not too far away from the object
            - on the ground
            - in the same room
        return reward, terminal, info
        """
        reached_goal = self.reached_goal(player_x, player_y, object_x, room_number)
        info['reached_goal'] = reached_goal
        if reached_goal:
            done = True
            reward = 1
        else:
            reward = 0  # override reward, such as when got key
        # terminate if agent enters another room
        in_same_room = room_number == self.room_number
        if not in_same_room:
            done = True
        # override needs_real_reset for EpisodicLifeEnv
        self.env.unwrapped.needs_real_reset = done or info.get("needs_reset", False)
        return reward, done, info
    