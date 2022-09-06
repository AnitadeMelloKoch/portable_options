from portable.environment import MonteAgentWrapper
from portable.utils import set_player_ram
import random

class MonteBootstrapWrapper(MonteAgentWrapper):
    def __init__(self, 
            env, 
            list_init_ram_states,
            list_init_states,
            list_init_agent_states,
            list_termination_points,
            reward_on_success=1,
            agent_space=False,
            max_steps=60 * 60 * 30):
        super().__init__(env, agent_space, stacked_observation=True, max_steps=max_steps)

        """
        env: initial environment to be wrapped
        list_init_ram_states: list of ram states that the agent can start at.
            when the environment is reset it will randomly select from these initiation
            states. These should be all the possible initiation states of the option in
            the first room.
        list_termination_points: list of tuples (player_x, player_y, room_number) which
            represent all the valid termination locations for the player
        """

        assert len(list_init_ram_states) > 0
        assert len(list_termination_points) > 0
        
        self.init_ram_states = list_init_ram_states
        self.init_states = list_init_states
        self.init_agent_states = list_init_agent_states
        self.termination_points = list_termination_points
        self.reward = reward_on_success

    def reset(self):
        rand_idx = random.randint(len(self.init_ram_states))
        ram = self.init_ram_states[rand_idx]
        s0 = set_player_ram(self.env, ram)
        self.stacked_agent_state = self.init_agent_states[rand_idx]
        self.stacked_state = self.init_states[rand_idx]
        self.num_lives = self.get_num_lives(self.get_current_ram())
        info = self.get_current_info(info={})
        self._elapsed_steps = 0

        player_x, player_y, _ = self.get_current_position()
        for _ in range(4):
            self.env.unwrapped.stacked_agent_position.append((player_x, player_y))
        
        if self.use_agent_space:
            if self.use_stacked_obs:
                s0 = self.stacked_agent_state
            else:
                s0 = self.agent_space()
        else:
            if self.use_stacked_obs:
                s0 = self.stacked_state
        
        return s0, info

    def _check_termination(self):
        player_x, player_y, room_num = self.get_current_position()

        if (player_x, player_y, room_num) in self.termination_points:
            return True

        return False

    def step(self, action):
        obs, reward, done, info = super().step(action)
        hit_termination = self._check_termination()
        if hit_termination and not done:
            done = True
            reward += self.reward
        
        return obs, reward, done, info
    
