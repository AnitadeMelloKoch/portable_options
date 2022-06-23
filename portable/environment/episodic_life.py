from gym import Wrapper


class EpisodicLifeEnv(Wrapper):
    def __init__(self, env):
        """
        Make end-of-life == end-of-episode, but only reset on true game end.

        basically the same as pfrl, but instead of using self.needs_real_reset,
        use self.env.unwrapped.needs_real_reset
        this is so that other wrappers can change how `done` is handled, such as MonteNewGoalWrapper

        NOTE: any wrapper that changes how done is handled should set self.env.unwrapped.needs_real_reset
        """
        super().__init__(env)
        self.lives = 0
        self.env.unwrapped.needs_real_reset = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.env.unwrapped.needs_real_reset = done or info.get("needs_reset", False)
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few
            # frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.env.unwrapped.needs_real_reset:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs
