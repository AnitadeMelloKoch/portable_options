from portable.environment.monte_object_goal_wrapper import MonteObjectGoalWrapper
from portable.option.option_utils import get_player_position, get_player_room_number


# y pos of the player in those rooms on the platform
room_to_y = {
    9: 235,
    11: 235,
    22: 235,
}

# for rooms that have two snakes, the x pos recorded here is the right snake
room_to_snake_x = {
    9: 51,
    11: 108,
    22: 31,
}


class MonteSnakeGoalWrapper(MonteObjectGoalWrapper):
    """
    for training a "jump over snake" skill.
    The agent finishes the skill if its y pos aligns with the floor of the snake and 
    its x pos is on the other side of the snake.

    currently, default to the player starts on the right side of the snake, and try to jump to the left of it
    """
    def __init__(self, env, epsilon_tol=8):
        super().__init__(env, epsilon_tol)
        self.y = room_to_y[self.room_number]
    
    def step(self, action):
        """
        override done and reward
        """
        next_state, reward, done, info = self.env.step(action)
        ram = self.env.unwrapped.ale.getRAM()
        room = get_player_room_number(ram)
        player_x, player_y = get_player_position(ram)
        snake_x = room_to_snake_x[self.room_number]
        reward, done = self.finished_skill(player_x, player_y, snake_x, room, done, info)
        return next_state, reward, done, info
