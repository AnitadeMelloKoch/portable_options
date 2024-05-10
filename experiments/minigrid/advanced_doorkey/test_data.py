from portable.option.memory import SetDataset
import matplotlib.pyplot as plt 
import numpy as np

initiation_positive_files = [
        # 'resources/minigrid_images/adv_doorkey_8x8_random_states_purpledoor.npy',
]

initiation_negative_files = [
        'resources/large_minigrid_images/adv_doorkey_getbluekey_doorblue_0_termination_negative.npy',
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

for _ in range(dataset_neg.num_batches):
    x, _ = dataset_neg.get_batch()
    for im in x:
        ax.imshow(np.transpose(im, (1,2,0)))
        plt.show(block=False)
        input("false")
