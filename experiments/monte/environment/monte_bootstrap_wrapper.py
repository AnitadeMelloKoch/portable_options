from experiments.monte.environment.agent_wrapper import MonteAgentWrapper
from portable.utils import set_player_ram
import random

class MonteBootstrapWrapper(MonteAgentWrapper):
    def __init__(self, 
            env, 
            list_init_states,
            list_termination_points,
            check_true_termination,
            reward_on_success=1,
            agent_space=False,
            max_steps=60 * 60 * 30
            ):
        super().__init__(env, agent_space, stack_observations=True, max_steps=max_steps)

        """
        env: initial environment to be wrapped
        list_init_ram_states: list of ram states that the agent can start at.
            when the environment is reset it will randomly select from these initiation
            states. These should be all the possible initiation states of the option in
            the first room.
        list_termination_points: list of tuples (player_x, player_y, room_number) which
            represent all the valid termination locations for the player
        """

        assert len(list_init_states) > 0
        assert len(list_termination_points) > 0
        
        self.init_states = list_init_states
        self.termination_points = list_termination_points
        self.reward = reward_on_success
        self.true_termination = []
        self._check_termination = check_true_termination
        
        self.obs = None

    def reset(self,
              agent_reposition_attempts=0,
              random_start=None,
              agent_position=None,
              return_rand_state_idx=False):
        self.env.reset()
        rand_idx = random.randint(0, len(self.init_states)-1)
        rand_state = self.init_states[rand_idx]
        self.true_termination = self.termination_points[rand_idx]
        s0 = set_player_ram(self.env, rand_state["ram"])
        self.stacked_agent_state = rand_state["agent_state"]
        self.stacked_state = rand_state["state"]
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
        
        if return_rand_state_idx is True:
            return s0, info, rand_idx
        
        self.obs = s0
        
        return s0, info

    def step(self, action):
        obs, reward, done, info = super().step(action)
        # remove environment reward
        # we only want to reward the agent for doing the option we are trying to train and nothing else
        reward = 0
        hit_termination = self._check_termination(self.get_current_position(), self.true_termination, self)
        if hit_termination and not done:
            done = True
            reward += self.reward
        
        self.obs = obs
        
        return obs, reward, done, info
    
    def get_term_state(self):
        return self.obs
    
