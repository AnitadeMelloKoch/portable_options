from portable.environment.monte_object_goal_wrapper import MonteObjectGoalWrapper
from portable.ale_utils import get_player_position, get_object_position, get_player_room_number


# y pos of the player in those rooms on the platform
room_to_y = {
    4: 235,
    13: 235, 
    21: 235,
}


class MonteSpiderGoalWrapper(MonteObjectGoalWrapper):
    """
    for training a "jump over spider" skill.
    The agent finishes the skill if its y pos aligns with the floor of the spider and 
    its x pos is on the other side of the spider.

    currently, default to the player starts on the right side of the spider, and try to jump to the left of it
    """
    def __init__(self, env, epsilon_tol=8, info_only=False):
        super().__init__(env, epsilon_tol, info_only)
        self.y = room_to_y[self.room_number]
    
    def step(self, action):
        """
        override done and reward
        """
        next_state, reward, done, info = self.env.step(action)
        ram = self.env.unwrapped.ale.getRAM()
        room = get_player_room_number(ram)
        player_x, player_y = get_player_position(ram)
        spider_x, _ = get_object_position(ram)
        goal_reward, terminal, info = self.finished_skill(player_x, player_y, spider_x, room, done, info)
        if not self.info_only:
            reward = goal_reward
            done = terminal
        return next_state, reward, done, info
