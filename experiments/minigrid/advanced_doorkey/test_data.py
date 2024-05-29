from portable.option.memory import SetDataset
import matplotlib.pyplot as plt 
import numpy as np
from pfrl.wrappers import atari_wrappers
from experiments.monte.environment import MonteBootstrapWrapper, MonteAgentWrapper

initiation_positive_files = [
        # 'resources/minigrid_images/adv_doorkey_8x8_random_states_purpledoor.npy',
]

initiation_negative_files = [
        'resources/monte_images/climb_down_ladder_room1_screen_termination_negative.npy',
]

dataset_pos = SetDataset(batchsize=16)
dataset_neg = SetDataset(batchsize=16)

fig = plt.figure(num=1, clear=True)
ax = fig.add_subplot()

# dataset_pos.add_true_files(initiation_positive_files)
dataset_neg.add_true_files(initiation_negative_files)

# for _ in range(dataset_pos.num_batches):
#         dataset_pos.shuffle()
#         x, _ = dataset_pos.get_batch()
#         for im in x:
#                 print(im.shape)
#                 print(np.transpose(im, (1,2,0)).shape)
#                 ax.imshow(np.transpose(im, (1,2,0)))
#                 plt.show(block=False)
#                 input("true")

env = atari_wrappers.wrap_deepmind(
        atari_wrappers.make_atari('MontezumaRevengeNoFrameskip-v4', max_frames=1000),
        episode_life=True,
        clip_rewards=True,
        frame_stack=False
)
env.seed(0)

env = MonteAgentWrapper(env, agent_space=False)

state, _ = env.reset()
# state,_,_,_ = env.step(0)
# state,_,_,_ = env.step(0)
# state,_,_,_ = env.step(0)
# state,_,_,_ = env.step(0)
# state,_,_,_ = env.step(0)
# print(state[3,60:80,60:80])

for _ in range(dataset_neg.num_batches):
    x, _ = dataset_neg.get_batch()
    for im in x:
        
        ax.imshow(np.transpose(im, (1,2,0)))
        plt.show(block=False)
        print(x.dtype)
        x = (x*255).int()
        print(x.dtype)
        print(x[0,3,60:80,60:80])
        input("false")
