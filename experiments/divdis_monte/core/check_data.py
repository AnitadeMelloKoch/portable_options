from portable.option.memory import SetDataset
import matplotlib.pyplot as plt 
import numpy as np

file = 'resources/monte_images/climb_down_ladder_room4_termination_negative.npy'

data = np.load(file)

print(data.shape)

for im in data:
    fig = plt.figure(num=1, clear=True)
    ax = fig.add_subplot()
    gs = fig.add_gridspec(nrows=1, ncols=4)
    for idx in range(4):
        ax = fig.add_subplot(gs[0,idx])
        ax.imshow(im[idx], cmap='gray')
        ax.axis('off')
    plt.show(block=False)
    input("continue")
            

# data = np.array(data)

# print(data[0])
# print(len(data))

# np.save(files[0], data)