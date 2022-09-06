from portable.environment.monte_object_goal_wrapper import MonteObjectGoalWrapper
from portable.utils import get_player_position, get_player_room_number, get_skull_position



room_to_skull_y = {
    1: 148,
    3: 235,
    5: 198,
    18: 235,
}


class MonteSkullGoalWrapper(MonteObjectGoalWrapper):
    """
    for training a "jump over skull" skill.
    The agent finishes the skill if its y pos aligns with the floor of the skull and 
    its x pos is on the other side of the skull.

    currently, default to the player starts on the right side of the skull, and try to jump to the left of it
    """
    def __init__(self, env, epsilon_tol=8, info_only=False):
        super().__init__(env, epsilon_tol, info_only)
        self.y = room_to_skull_y[self.room_number]
    
    def step(self, action):
        """
        override done and reward
        """
        next_state, reward, done, info = self.env.step(action)
        ram = self.env.unwrapped.ale.getRAM()
        room = get_player_room_number(ram)
        player_x, player_y = get_player_position(ram)
        skull_x = get_skull_position(ram)
        goal_reward, terminal, info = self.finished_skill(player_x, player_y, skull_x, room, done, info)
        if not self.info_only:
            reward = goal_reward
            done = terminal
        return next_state, reward, done, info
