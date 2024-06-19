def monte_with_skills(game_name=None,
                      sticky_actions=True,
                      episode_life=True,
                      clip_rewards=False,
                      frame_skip=4,
                      frame_stack=4,
                      frame_warp=(84, 84),
                      max_episode_steps=None,
                      single_life=False,
                      single_screen=False,
                      seed=None,
                      noop_wrapper=False,
                      render_option_execution=False):
    from gym_montezuma.envs.montezuma_env import make_monte_env_as_atari_deepmind

    d = {
        "single_life": single_life,
        "single_screen": single_screen,
        "seed": seed,
        "noop_wrapper": noop_wrapper,
        "render_option_execution": render_option_execution
    }

    env = make_monte_env_as_atari_deepmind(max_episode_steps=max_episode_steps,
                                           episode_life=episode_life,
                                           clip_rewards=clip_rewards,
                                           frame_skip=frame_skip,
                                           frame_stack=frame_stack,
                                           frame_warp=frame_warp,
                                           **d)
    return env

env = monte_with_skills(render_option_execution=True)

state = env.reset()
print(state.numpy())