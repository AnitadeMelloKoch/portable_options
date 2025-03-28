import numpy as np
import matplotlib.pyplot as plt 

data = np.load(f'resources/sokoban_images/box_1_0_positive.npy')

print(data.shape)

for img in data:
    plt.imshow(img)
    plt.show(block=False)
    plt.pause(0.2)
    plt.cla()

