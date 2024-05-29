from portable.utils import set_player_ram
from portable.utils import load_init_states
from experiments.monte.environment import MonteBootstrapWrapper
from experiments.monte.environment import MonteAgentWrapper
from pfrl.wrappers import atari_wrappers
import matplotlib.pyplot as plt

env = atari_wrappers.wrap_deepmind(
    atari_wrappers.make_atari('MontezumaRevengeNoFrameskip-v4', max_frames=1000),
        episode_life=True,
        clip_rewards=True,
        frame_stack=False
    )
env = MonteAgentWrapper(env, agent_space=False)

ram_dict = load_init_states(["resources/monte_env_states/room1/enemy/skull_right_1.pkl"])[0]

env.reset()

set_player_ram(env, ram_dict["ram"])

print(env.get_current_info({})["player_pos"])

img = env.render("rgb_array")
plt.imshow(img)
plt.show()


