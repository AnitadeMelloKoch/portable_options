from portable.option.memory import SetDataset
import matplotlib.pyplot as plt 
import numpy as np
from pfrl.wrappers import atari_wrappers
from experiments.monte.environment import MonteBootstrapWrapper, MonteAgentWrapper



files = [
        'resources/large_minigrid/adv_doorkey_16x16_v2_togoal_1_2_termination_negative.npy',
]

dataset = SetDataset(batchsize=16)

fig = plt.figure(num=1, clear=True)
ax = fig.add_subplot()

dataset.add_true_files(files)

for _ in range(dataset.num_batches):
        dataset.shuffle()
        x, _ = dataset.get_batch()
        for im in x:
                print(im.shape)
                print(np.transpose(im, (1,2,0)).shape)
                ax.imshow(np.transpose(im, (1,2,0)))
                fig.savefig("test_image.png")
                input("continue")

# gs = fig.add_gridspec(nrows=1, ncols=4)
# for _ in range(dataset.num_batches):
#         x, _ = dataset.get_batch()
#         for im in x:
#                 for idx in range(4):
#                         ax = fig.add_subplot(gs[0,idx])
#                         ax.imshow(im[idx], cmap='gray')
#                         ax.axis('off')
#                 plt.show(block=False)
#                 input("true")